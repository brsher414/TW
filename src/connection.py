import csv
import os
import time

import oracledb
from oracledb.exceptions import InterfaceError
from src.logger import get_logger

logger = get_logger(__name__)
DEFAULT_FETCH_SIZE = int(os.getenv("ORACLE_FETCH_SIZE", "10000"))


def create_oracle_connection_pool(service="02") -> oracledb.ConnectionPool:
    """Initialize Oracle client and create a connection pool."""
    logger.info(f"Initializing Oracle connection pool for service: {service}")
    try:
        if service == "02":
            service_name = os.environ["SERVICE_NAME_02"]
        elif service == "03":
            service_name = os.environ["SERVICE_NAME_03"]
        else:
            raise ValueError(f"Unsupported service: {service}")

        oracledb.init_oracle_client()
        pool = oracledb.create_pool(
            user=os.environ["ORACLE_USER"],
            password=os.environ["ORACLE_PSWD"],
            dsn=os.environ["ORACLE_HOST"] + ":" + str(1521) + "/" + service_name,
            min=2,
            max=30,
        )
        logger.info("Oracle connection pool created successfully.")
        return pool
    except Exception as e:
        logger.error(f"Failed to create Oracle connection pool: {e}")
        raise


def stream_query_to_csv(
    pool,
    sql,
    output_path,
    params=None,
    fetch_size: int | None = None,
    query_timeout_ms: int | None = None,
) -> int:
    """Stream query rows directly to a CSV file without accumulating the full result set in memory."""

    def process_lob(value):
        return value.read() if isinstance(value, oracledb.LOB) else value

    def is_write_operation(query):
        keywords = ("INSERT", "UPDATE", "DELETE")
        return any(keyword in query.strip().upper() for keyword in keywords)

    def is_ddl_operation(query):
        keywords = ("CREATE", "ALTER", "TRUNCATE", "DROP")
        return any(keyword in query.strip().upper() for keyword in keywords)

    def get_fetch_size() -> int:
        try:
            resolved = int(fetch_size) if fetch_size is not None else DEFAULT_FETCH_SIZE
        except (TypeError, ValueError):
            resolved = DEFAULT_FETCH_SIZE
        return max(1, resolved)

    try:
        with pool.acquire() as conn, conn.cursor() as cursor:
            effective_fetch_size = get_fetch_size()
            cursor.arraysize = effective_fetch_size
            if query_timeout_ms is not None and query_timeout_ms > 0:
                conn.call_timeout = int(query_timeout_ms)
                cursor.call_timeout = int(query_timeout_ms)
            logger.info("stream_query_to_csv executing SQL (fetch_size=%s, timeout_ms=%s)", effective_fetch_size, query_timeout_ms)
            start_execute = time.perf_counter()
            cursor.execute(sql, params or {})
            logger.info("stream_query_to_csv SQL execute finished in %.2fs", time.perf_counter() - start_execute)

            if is_write_operation(sql) or is_ddl_operation(sql):
                conn.commit()
                return 0

            with open(output_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                header = [desc[0] for desc in cursor.description] if cursor.description else []
                if header:
                    writer.writerow(header)

                total_rows = 0
                fetch_logged = False
                while True:
                    fetch_start = time.perf_counter()
                    rows = cursor.fetchmany(effective_fetch_size)
                    if not fetch_logged:
                        logger.info("stream_query_to_csv first fetch finished in %.2fs", time.perf_counter() - fetch_start)
                        fetch_logged = True
                    if not rows:
                        break
                    for row in rows:
                        writer.writerow(tuple(process_lob(value) for value in row))
                        total_rows += 1

            return total_rows
    except Exception as exc:
        logger.error("Error streaming query to CSV: %s", exc)
        raise


def execute_query(
    pool,
    sql,
    params=None,
    return_header=False,
    fetch_size: int | None = None,
) -> list[tuple] | tuple[list[tuple], list[str]]:
    """Execute a SQL query using the provided connection pool.

    Args:
        pool: The connection pool to use.
        sql: The SQL query to execute.
        params: Optional parameters for the SQL query.
        return_header: Whether to return the query result header.
        fetch_size: Rows fetched per batch for read queries. Defaults to
            ORACLE_FETCH_SIZE env var (fallback 10000).

    Returns:
        The result of the query as a list of tuples, optionally with the header.
    """

    def process_lob(value):
        """Process LOB values to read their content."""
        #isinsatance的两个参数分别是value,检查对象，oracledb.LOB,检查类型。 如果value是oracledb.LOB的实例，就调用read()方法读取内容，否则直接返回value。
        return value.read() if isinstance(value, oracledb.LOB) else value

    def is_write_operation(query):
        """Check if the query is a write operation."""
        keywords = ("INSERT", "UPDATE", "DELETE")
        return any(keyword in query.strip().upper() for keyword in keywords)
    
    def is_ddl_operation(query):
        """20260325 - Check if the query is a DDL operation."""
        keywords = ("CREATE", "ALTER", "TRUNCATE", "DROP")
        return any(keyword in query.strip().upper() for keyword in keywords)

    def get_fetch_size() -> int:
        """Resolve effective fetch size with a safe lower bound."""
        try:
            resolved = int(fetch_size) if fetch_size is not None else DEFAULT_FETCH_SIZE
        #为什么except能有这种写法？ 不都是except冒号吗？ 括号是什么意思？
        except (TypeError, ValueError): 
            resolved = DEFAULT_FETCH_SIZE
        return max(1, resolved)

    try:
        with pool.acquire() as conn, conn.cursor() as cursor:
            effective_fetch_size = get_fetch_size()
            cursor.arraysize = effective_fetch_size
            cursor.execute(sql, params or {})

            if is_write_operation(sql) or is_ddl_operation(sql):
                conn.commit()
                logger.info("Transaction committed.")
                return []

            result: list[tuple] = []
            while True:
                rows = cursor.fetchmany(effective_fetch_size)
                if not rows:
                    break
                result.extend(
                    tuple(process_lob(value) for value in row) for row in rows
                )

            header = (
                [desc[0] for desc in cursor.description] if cursor.description else []
            )

            return (result, header) if return_header else result
    except InterfaceError as e:
        if "1003" in str(e):
            logger.warning("No results returned for the query.")
            return []
        logger.error(f"Database interface error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise
