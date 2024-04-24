import warnings
warnings.filterwarnings("ignore") 
from common_utils import *
from pathlib import Path
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor


def build_nuboard(scenario_builder, simulation_path):
    nuboard = NuBoard(
        nuboard_paths=simulation_path,
        scenario_builder=scenario_builder,
        vehicle_parameters=get_pacifica_parameters(),
        port_number=5012
    )
    nuboard.run()

def find_nuboard_files(root_dir):
    """Find all .nuboard files within root_dir and its subdirectories."""
    root_path = Path(root_dir)
    nuboard_files = list(root_path.rglob('*.nuboard'))
    return [str(file) for file in nuboard_files]

# build scenarios
print('Extracting scenarios...')
data_root = 'nuplan/dataset/nuplan-v1.1/splits/mini'
map_root = 'nuplan/dataset/maps'

scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
builder = NuPlanScenarioBuilder(data_root, map_root, sensor_root=None, db_files=None, map_version="nuplan-maps-v1.0", scenario_mapping=scenario_mapping)

# scenario_filter = ScenarioFilter(*get_filter_parameters(10, None, False))
# worker = SingleMachineParallelExecutor(use_process_pool=True)
# scenarios = builder.get_scenarios(scenario_filter, worker)
# del worker, scenario_filter, scenario_mapping

root_directory = 'testing_log/open_loop_boxes/gameformer_planner'  # Replace with your root directory path
simulation_file = find_nuboard_files(root_directory)

build_nuboard(builder, simulation_file) 
