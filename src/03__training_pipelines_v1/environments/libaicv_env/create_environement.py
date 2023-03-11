from azureml.core import Workspace,Environment
from azureml.core.conda_dependencies import CondaDependencies

BUILD = True

ws=Workspace.from_config()
# We get the existing base environment
new_env = Environment.from_dockerfile(name="libaicv_env",dockerfile="environments/libaicv_env/Dockerfile")
# We add our libaicv library and get the link.
library_url = Environment.add_private_pip_wheel(
    workspace=ws,
    file_path = "environments\libaicv_env\library\libaicv-0.1.0-py3-none-any.whl",
    exist_ok=True)
# We get the token SaS from destination container...
token_sas = "<TOKEN SAS>"
print('Library URL to add to your Dockerfile before build')
print(f">> {library_url+token_sas}")
#
# Register the environment
if BUILD:
    new_env.register(workspace=ws)
    new_env.build(workspace=ws)