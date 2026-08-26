import json
from core.project_context import ProjectContext
from core.run_manifest import create_manifest,set_active_run
c=ProjectContext.active(project_root=__import__('pathlib').Path(__file__).resolve().parent);c.ensure_directories();create_manifest(c);set_active_run(c);print(json.dumps({"category":c.category_code,"run_id":c.run_id,"run_dir":str(c.run_dir),"etl_output":str(c.sampled_products_file)},ensure_ascii=False,indent=2))
