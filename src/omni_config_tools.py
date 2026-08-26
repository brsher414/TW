import os
import re

import numpy as np
import oracledb
import pandas as pd
from .logger import get_logger

logger = get_logger(__name__)
DEFAULT_OMNI_FETCH_SIZE = int(
    os.getenv("OMNI_CONFIG_FETCH_SIZE", os.getenv("ORACLE_FETCH_SIZE", "10000"))
)

OMNI_CONIG_TABLES = [
    "omni_db_cate_segment",
    "omni_db_dic_segment",
    "omni_special_logic",
    "omni_special_regular",
    "omni_special_param_global",
    "db_segment_mapping_v40",
    "omni_db_cate_rule",
]


class OracleConfigModifier:
    def __init__(self, pool: oracledb.ConnectionPool, suffix: str):
        self.pool = pool
        if not re.match(r"^[a-zA-Z]+$", suffix):
            raise ValueError("Suffix must contain only alpha characters ")
        self.suffix = suffix

    def create_or_reset_table(self):
        with self.pool.acquire() as conn, conn.cursor() as cursor:
            cursor.execute(
                f"""
                CALL ECOM.CREATE_OMNI_TABLES_PROC('{self.suffix}')
                """
            )
            try:
                cursor.execute(
                    f"""
                    CREATE TABLE DB_SEGMENT_MAPPING_V40_{self.suffix} AS SELECT * FROM DB_SEGMENT_MAPPING_V40
                    """
                )
            except oracledb.DatabaseError as e:
                if "ORA-00955" in str(e):
                    cursor.execute(
                        f"""
                        TRUNCATE TABLE DB_SEGMENT_MAPPING_V40_{self.suffix}
                        """
                    )
                    cursor.execute(
                        f"""
                        INSERT INTO DB_SEGMENT_MAPPING_V40_{self.suffix}
                        SELECT * FROM DB_SEGMENT_MAPPING_V40
                        """
                    )
                    conn.commit()

    def insert_data_to_table(self, traget_table: str, data: list[dict]):
        cols = data[0].keys() if data else []
        sql = f"""
        insert into {traget_table} ({", ".join(cols)})
        values ({", ".join([f":{col}" for col in cols])})
        """
        try:
            with self.pool.acquire() as conn:
                with conn.cursor() as cursor:
                    cursor.executemany(sql, data)
                conn.commit()

            logger.info("Inserted %d records into %s", len(data), traget_table)
            return len(data)

        except Exception as e:
            # Log full context for debugging and re-raise for UI layer to handle
            logger.exception(
                "Failed to insert into %s. Error: %s. SQL: %s. Data sample: %s",
                traget_table,
                str(e),
                sql,
                data[:1] if data else None,
            )
            raise

    def update_data_to_table(
        self, target_table: str, update_on: list[str], data: list[dict]
    ):
        data_cols = data[0].keys() if data else []
        sql = f"""
        update {target_table}
        set {", ".join([f"{col} = :{self.sanitize(col)}" for col in data_cols if col not in update_on])}
        where {" and ".join([f"{col} = :{self.sanitize(col)}" for col in update_on])}
        """
        sanitized_data = [{self.sanitize(k): v for k, v in row.items()} for row in data]
        with self.pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, sanitized_data)
            conn.commit()

        return True

    def sanitize(self, col):
        return "RID" if col.upper() == "ROWID" else col

    def delete_data_from_table(
        self, target_table: str, delete_on: list[str], data: list[dict]
    ):
        sql = f"""
        delete from {target_table}
        where {" and ".join([f"{col} = :{self.sanitize(col)}" for col in delete_on])}
        """

        sanitized_data = [{self.sanitize(k): v for k, v in row.items()} for row in data]
        try:
            with self.pool.acquire() as conn:
                with conn.cursor() as cursor:
                    cursor.executemany(sql, parameters=sanitized_data)
                conn.commit()
            logger.info("Deleted %d records from %s", len(sanitized_data), target_table)
            return len(sanitized_data)
        except Exception:
            logger.exception(
                "Failed to delete records from %s. SQL: %s", target_table, sql
            )
            raise


class ConfigTools:
    config_base_name: str = ""
    prikey: tuple[str] = tuple()

    def __init__(self, pool: oracledb.ConnectionPool, suffix: str = "WG"):
        self.pool = pool
        self.suffix = suffix
        self.config_name = f"{self.config_base_name}_{suffix}"
        self.config_modifier = OracleConfigModifier(pool, suffix)
        self.schema = (
            self.get_table_schema()
            if "ROWID" not in self.prikey
            else pd.concat(
                [
                    self.get_table_schema(),
                    pd.DataFrame(
                        [
                            {
                                "column_name": "ROWID",
                                "data_type": "ROWID",
                                "python_type": str,
                            }
                        ]
                    ),
                ]
            )
        )

    def _resolve_fetch_size(self) -> int:
        """Return effective batch size used by fetchmany."""
        try:
            size = int(DEFAULT_OMNI_FETCH_SIZE)
        except (TypeError, ValueError):
            size = 10000
        return max(1, size)

    def _fetch_all_rows(self, cursor) -> list[tuple]:
        """Read cursor rows in batches to avoid single huge fetchall call."""
        batch_size = self._resolve_fetch_size()
        cursor.arraysize = batch_size

        records: list[tuple] = []
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            records.extend(rows)
        return records

    def apply_df_to_config(
        self,
        df: pd.DataFrame,
        existing_keys: set | None = None,
    ) -> bool:
        """
        Apply the DataFrame to the Oracle configuration.
        """
        if df.empty:
            return
        df = self.process_df(df)

        if "delete" in df.columns:
            delete_mark = df["delete"] == 1
            self._delete_marked_records(df[delete_mark])
            df = df[~delete_mark]

        if existing_keys is None:
            existing_keys = self._get_existing_keys()
        self._insert_new_records(df, existing_keys)
        self._update_existing_records(df, existing_keys)

        return True

    def apply_df_to_config_with_version(
        self,
        df: pd.DataFrame,
        request_no: str
    ) -> bool:
        """
        Apply the DataFrame to the Oracle configuration with version.
        """
        if df.empty:
            return
        
        df['REQUEST_NO'] = request_no
        
        df = self.process_df(df)

        self._insert_new_records(df, set())
        return True

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the DataFrame according to the specific configuration.
        This method should be overridden in subclasses.
        """
        cols = tuple(self.schema["column_name"])
        available_cols = list(cols) + list(["NEW_" + col for col in cols]) + ["delete"]

        type_map = {
            k: v
            for k, v in self.schema.loc[:, ["column_name", "python_type"]].itertuples(
                index=False, name=None
            )
            if k in df.columns
        }
        type_map.update(
            {"NEW_" + k: v for k, v in type_map.items() if "NEW_" + k in df.columns}
        )
        return (
            df.drop([col for col in df.columns if col not in available_cols], axis=1)
            .replace(np.nan, None)
            .astype(type_map, errors="ignore")  # 没有ignore会报错
            .replace("None", None)  # Ensure None is used for null values
        )

    def _get_existing_keys(self) -> set:
        """Get existing composite keys from the configuration"""
        if not self.prikey:
            return set()

        sql = f"SELECT {','.join(col for col in self.prikey)} FROM {self.config_name}"
        with self.pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                records = self._fetch_all_rows(cursor)

        return {tuple(str(value) for value in row) for row in records}

    def _insert_new_records(self, df: pd.DataFrame, existing_keys: set) -> None:
        """Insert new records that don't exist in the current configuration"""
        # Filter columns to only those in the schema
        cols = [col for col in self.schema["column_name"] if col in df.columns]
        new_keys = df.loc[:, self.prikey].astype(str).apply(tuple, axis=1)
        df_need_insert = df.loc[~new_keys.isin(existing_keys), cols].filter(
            regex="^(?!NEW_).*$", axis=1
        )

        cols_need_insert = [col for col in cols if col != "ROWID"]

        if not df_need_insert.empty:
            self.insert_config(
                df_need_insert.loc[:, cols_need_insert].to_dict(orient="records")
            )

    def _update_existing_records(
        self, df: pd.DataFrame, existing_keys: set
    ) -> None:
        """Update existing records with new values"""
        existing_mask = (
            df.loc[:, self.prikey].astype(str).apply(tuple, axis=1).isin(existing_keys)
        )
        df_need_update: pd.DataFrame = df.loc[existing_mask]

        for col in df_need_update.columns:
            if col.startswith("NEW_"):
                self._process_single_column_update(df_need_update, col)

    def _process_single_column_update(
        self, df_need_update: pd.DataFrame, col: str
    ) -> None:
        """Process updates for a single NEW_* column"""
        update_df: pd.DataFrame = df_need_update.loc[
            ~df_need_update[col].isna(), list(self.prikey) + [col]
        ].rename(columns={col: col.replace("NEW_", "")})

        if not update_df.empty:
            self.update_config(update_df.to_dict(orient="records"))

    def _delete_marked_records(self, df: pd.DataFrame) -> None:
        """Delete records marked for deletion"""
        if "delete" not in df.columns:
            return
        # st.dataframe(df)
        df_need_delete: pd.DataFrame = df.loc[df["delete"] == 1, self.prikey]

        if not df_need_delete.empty:
            self.delete_config(df_need_delete.to_dict(orient="records"))

    def get_prikey_df(self) -> pd.DataFrame:
        """
        Get the existing configuration DataFrame from the Oracle database.
        """
        sql = f"SELECT {','.join(col for col in self.prikey)} FROM {self.config_name}"
        with self.pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                records = self._fetch_all_rows(cursor)
        df_config = pd.DataFrame(records, columns=self.prikey, dtype=str)
        return df_config

    # def get_unikey_df(self) -> pd.DataFrame:
    #     """
    #     Get the existing configuration DataFrame from the Oracle database.
    #     This method retrieves all columns except the primary key columns.
    #     """
    #     sql = f"SELECT {','.join(col for col in self.unikey)} FROM {self.config_name}"
    #     with self.pool.acquire() as conn:
    #         with conn.cursor() as cursor:
    #             cursor.execute(sql)
    #             records = cursor.fetchall()
    #     df_config = pd.DataFrame(records, columns=self.unikey)
    #     return df_config

    def update_config(self, records: list[dict]):
        """
        Update the existing configuration in the Oracle database.
        """
        if not self.prikey:
            raise ValueError("No update columns specified.")
        self.config_modifier.update_data_to_table(
            self.config_name, self.prikey, records
        )

    def delete_config(self, records: list[dict]):
        """
        Delete records from the existing configuration in the Oracle database.
        """
        if not self.prikey:
            raise ValueError("No delete columns specified.")
        self.config_modifier.delete_data_from_table(
            self.config_name, self.prikey, records
        )

    def insert_config(self, records: list[dict]):
        """
        Insert new records into the existing configuration in the Oracle database.
        """
        self.config_modifier.insert_data_to_table(self.config_name, records)

    def get_table_schema(self) -> pd.DataFrame:
        """
        Get the schema of the configuration table.
        """
        sql = f"""
        SELECT column_name, data_type
        FROM user_tab_columns
        WHERE table_name = UPPER('{self.config_name}')
        """
        with self.pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                schema_records = self._fetch_all_rows(cursor)

        df_schema = pd.DataFrame(
            schema_records, columns=["column_name", "data_type"]
        ).loc[
            lambda df: ~df["column_name"].isin(
                ("LAST_UPDATE_TIME", "OPERATOR", "INSERTDATE")
            )
        ]
        type_map = {
            "VARCHAR2": str,
            "NVARCHAR2": str,
            "NUMBER": int,
            "DATE": pd.Timestamp,
            "CHAR": str,
            "CLOB": str,
            "BLOB": bytes,
        }
        df_schema["python_type"] = df_schema["data_type"].map(type_map).fillna(object)

        return df_schema


class CateSegmentTools(ConfigTools):
    config_base_name = "omni_db_cate_segment"
    prikey = ("CATCODE", "SEGNO")


class DicSegmentTools(ConfigTools):
    config_base_name = "omni_db_dic_segment"
    prikey = ("SEGID",)

    def apply_df_to_config(
        self,
        df: pd.DataFrame,
    ) -> bool:
        # Assign available SEGID to rows that need to be inserted

        def is_numeric(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def convert_to_str_if_numeric(value):
            if pd.isna(value):
                return value
            if isinstance(value, str) and is_numeric(value):
                return str(int(float(value)))
            elif isinstance(value, (int, float)):
                return str(int(value))
            return value

        df.loc[:, "SEGID"] = df["SEGID"].apply(convert_to_str_if_numeric)
        df.loc[:, "SEGNO"] = df["SEGNO"].apply(convert_to_str_if_numeric)
        df.loc[:, "ORD"] = df["ORD"].apply(convert_to_str_if_numeric)
        df.loc[:, "FNO"] = df["FNO"].apply(convert_to_str_if_numeric)

        for col in ["NEW_ORD", "NEW_FNO"]:
            if col in df.columns:
                df.loc[:, col] = df[col].apply(convert_to_str_if_numeric)

        # df.loc[~df["ORD"].isna(), "ORD"] = (
        #     df.loc[~df["ORD"].isna(), "ORD"].astype(int).astype(str)
        # )
        # df.loc[~df["SEGID"].isna(), "SEGID"] = (
        #     df.loc[~df["SEGID"].isna(), "SEGID"].astype(int).astype(str)
        # )

        # # 先将 FNO 列转换为字符串，避免 is_numeric 报错
        # df["FNO"] = df["FNO"].astype(str)

        # # 使用 str.isnumeric() 筛选出纯数字字符串
        # mask = df["FNO"].str.isnumeric()

        # # 对这些值进行转换：先转 int，再转回 str
        # df.loc[mask, "FNO"] = df.loc[mask, "FNO"].astype(int).astype(str)

        # import streamlit as st

        # st.dataframe(
        #     df.assign(
        #         need_insert=lambda df: ~df.loc[:, self.prikey]
        #         .astype(str)
        #         .apply(tuple, axis=1)
        #         .isin(
        #             self.get_prikey_df()
        #             .loc[:, self.prikey]
        #             .astype(str)
        #             .apply(tuple, axis=1)
        #         )
        #     )
        # )

        existing_keys = self._get_existing_keys()
        incoming_keys = df.loc[:, self.prikey].astype(str).apply(tuple, axis=1)
        df_need_insert = df.loc[
            ((~incoming_keys.isin(existing_keys)) | df.loc[:, self.prikey].isna().any(axis=1)),
            :,
        ].copy()

        if not df_need_insert.empty:
            available_segid = self.get_available_segid(len(df_need_insert))
            df_need_insert["SEGID"] = list(available_segid)

        # Combine the updated DataFrame with the original DataFrame
        df = pd.concat([df.loc[~df.index.isin(df_need_insert.index)], df_need_insert])
        # import streamlit as st

        # st.dataframe(df)

        return super().apply_df_to_config(df, existing_keys=existing_keys)

    def apply_df_to_config_with_version(
        self,
        df: pd.DataFrame,
        request_no: str
    ) -> bool:
        # Assign available SEGID to rows that need to be inserted

        def is_numeric(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def convert_to_str_if_numeric(value):
            if pd.isna(value):
                return value
            if isinstance(value, str) and is_numeric(value):
                return str(int(float(value)))
            elif isinstance(value, (int, float)):
                return str(int(value))
            return value

        df.loc[:, "SEGID"] = df["SEGID"].apply(convert_to_str_if_numeric)
        df.loc[:, "SEGNO"] = df["SEGNO"].apply(convert_to_str_if_numeric)
        df.loc[:, "ORD"] = df["ORD"].apply(convert_to_str_if_numeric)
        df.loc[:, "FNO"] = df["FNO"].apply(convert_to_str_if_numeric)

        for col in ["NEW_ORD", "NEW_FNO"]:
            if col in df.columns:
                df.loc[:, col] = df[col].apply(convert_to_str_if_numeric)

        # df.loc[~df["ORD"].isna(), "ORD"] = (
        #     df.loc[~df["ORD"].isna(), "ORD"].astype(int).astype(str)
        # )
        # df.loc[~df["SEGID"].isna(), "SEGID"] = (
        #     df.loc[~df["SEGID"].isna(), "SEGID"].astype(int).astype(str)
        # )

        # # 先将 FNO 列转换为字符串，避免 is_numeric 报错
        # df["FNO"] = df["FNO"].astype(str)

        # # 使用 str.isnumeric() 筛选出纯数字字符串
        # mask = df["FNO"].str.isnumeric()

        # # 对这些值进行转换：先转 int，再转回 str
        # df.loc[mask, "FNO"] = df.loc[mask, "FNO"].astype(int).astype(str)

        # import streamlit as st

        # st.dataframe(
        #     df.assign(
        #         need_insert=lambda df: ~df.loc[:, self.prikey]
        #         .astype(str)
        #         .apply(tuple, axis=1)
        #         .isin(
        #             self.get_prikey_df()
        #             .loc[:, self.prikey]
        #             .astype(str)
        #             .apply(tuple, axis=1)
        #         )
        #     )
        # )

        existing_keys = self._get_existing_keys()
        incoming_keys = df.loc[:, self.prikey].astype(str).apply(tuple, axis=1)
        df_need_insert = df.loc[
            ((~incoming_keys.isin(existing_keys)) | df.loc[:, self.prikey].isna().any(axis=1)),
            :,
        ].copy()

        if not df_need_insert.empty:
            available_segid = self.get_available_segid(len(df_need_insert))
            df_need_insert["SEGID"] = list(available_segid)

        # Combine the updated DataFrame with the original DataFrame
        df = pd.concat([df.loc[~df.index.isin(df_need_insert.index)], df_need_insert])
        
        return super().apply_df_to_config_with_version(df, request_no)

    def get_available_segid(
        self,
        num_segid: int = 1000,
    ):
        df_segid = (
            (
                self.get_prikey_df()
                .astype({"SEGID": int})
                .loc[lambda df: df["SEGID"] > 0]
                .sort_values(by="SEGID")
                .assign(
                    start=lambda x: x["SEGID"] + 1,
                    end=lambda x: x["SEGID"].shift(-1),
                    diff=lambda x: x["SEGID"].shift(-1) - x["SEGID"] - 1,
                )
                .assign(cum_diff=lambda x: x["diff"].cumsum())
            )
            .fillna(0)
            .astype(int)
            .loc[lambda df: df["diff"] > 0]
        )

        min_segid = df_segid.loc[lambda df: df["cum_diff"] >= num_segid, "SEGID"].min()

        segid_records = df_segid.loc[lambda df: df["SEGID"] <= min_segid].to_dict(
            "records"
        )

        def _get_segid_from_records(records):
            for record in records:
                start, end = record["start"], record["end"]
                for segid in range(start, end):
                    yield segid

        cnt = 0
        while cnt < num_segid:
            for segid in _get_segid_from_records(segid_records):
                yield segid
                cnt += 1
                if cnt >= num_segid:
                    break


class SpecialLogicTools(ConfigTools):
    config_base_name = "omni_special_logic"
    prikey = ("WORD1", "TYPE", "CATEGORY")


class SpecialRegularTools(ConfigTools):
    config_base_name = "omni_special_regular"
    prikey = ("CATEGORY", "SEGNO", "PATTERN")


class SpecialParamGlobalTools(ConfigTools):
    config_base_name = "omni_special_param_global"
    prikey = ("SEGNO", "CATEGORY", "TYPE")


class SegmentMappingTools(ConfigTools):
    config_base_name = "db_segment_mapping_v40"
    prikey = ("ROWID",)

    def apply_df_to_config(self, df):
        int_cols = ["ATTRNO1", "ATTRNO2", "ATTRNO3", "ATTRNO4", "R_ATTRNO"]
        for col in int_cols:
            df.loc[:, col] = df[col].astype(int, errors="ignore")

        return super().apply_df_to_config(df)
    
    
    def apply_df_to_config_with_version(self, df, request_no):
        int_cols = ["ATTRNO1", "ATTRNO2", "ATTRNO3", "ATTRNO4", "R_ATTRNO"]
        for col in int_cols:
            df.loc[:, col] = df[col].astype(int, errors="ignore")

        return super().apply_df_to_config_with_version(df, request_no)

class CategoryTools(ConfigTools):
    config_base_name = "OMNI_DB_CATE_RULE"
    prikey = ("ROWID",)

TOOL_MAP = {
    "omni_db_dic_segment": DicSegmentTools,
    "omni_db_cate_segment": CateSegmentTools,
    "omni_special_logic": SpecialLogicTools,
    "omni_special_regular": SpecialRegularTools,
    "omni_special_param_global": SpecialParamGlobalTools,
    "db_segment_mapping_v40": SegmentMappingTools,
    "omni_db_cate_rule": CategoryTools,
}