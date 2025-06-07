import matplotlib.pyplot as plt
import seaborn as sb
import xarray as xr
import pymc as pm
import numpy as np
import graphviz
import arviz as az
import pytensor as pt
from typing import Optional, Tuple, Set, List, Dict, Union, Callable
from dataclasses import dataclass
import simpy as sim
import logging
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode="w",
    filename="process_analysis.log"
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessGraph:

    @staticmethod
    def serial_data_dev() -> graphviz.Digraph:
        """
        Creates a directed graph representing the data development process
        and its dependencies using graphviz.

        Returns:
            graphviz.Digraph: A graphviz Digraph object representing the graph.
        """
        # Create a directed graph object
        dot = graphviz.Digraph(comment='Software Development Process', format='png')  # You can change format if needed

        # Add nodes for each stage
        dot.node('Requirements', 'Requirements\nSpecification')
        dot.node('Enrichment', 'Enrichment')
        dot.node('Modeling', 'Modeling')
        dot.node('Hydration', 'Hydration')
        dot.node('Testing', 'Testing')
        dot.node('Deployment', 'Deployment\n&\nRelease')

        # Add edges to represent dependencies for a serial process
        dot.edge('Requirements', 'Enrichment')
        dot.edge('Enrichment', 'Modeling')
        dot.edge('Modeling', 'Hydration')
        dot.edge('Hydration', 'Testing')
        dot.edge('Testing', 'Deployment')

        # You can customize the graph appearance (optional)
        dot.attr('node', shape='box', style='rounded', fontname='Arial', fontsize='12')
        dot.attr('edge', arrowtail='empty', arrowhead='normal', color='darkgreen')
        dot.attr(size='8.5,11')  # Set the size of the output graph (optional)

        return dot

    @staticmethod
    def serial_software_dev() -> graphviz.Digraph:
        """
        Creates a directed graph representing the software development process
        and its dependencies using graphviz.

        Returns:
            graphviz.Digraph: A graphviz Digraph object representing the graph.
        """
        # Create a directed graph object
        dot = graphviz.Digraph(comment='Software Development Process', format='png')  # You can change format if needed

        # Add nodes for each stage
        dot.node('Requirements', 'Requirements\nSpecification')
        dot.node('Design', 'UI/UX Design')
        dot.node('Backend', 'Backend\nDevelopment')
        dot.node('Frontend', 'Frontend\nDevelopment')
        dot.node('Integration', 'Integration\n&\nTesting')
        dot.node('Deployment', 'Deployment\n&\nRelease')

        # Add edges to represent dependencies for a serial process
        dot.edge('Requirements', 'Design')
        dot.edge('Design', 'Backend')
        dot.edge('Backend', 'Frontend')
        dot.edge('Frontend', 'Integration')
        dot.edge('Integration', 'Deployment')

        # You can customize the graph appearance (optional)
        dot.attr('node', shape='box', style='rounded', fontname='Arial', fontsize='12')
        dot.attr('edge', arrowtail='empty', arrowhead='normal', color='darkgreen')
        dot.attr(size='8.5,11')  # Set the size of the output graph (optional)

        return dot

    @staticmethod
    def parallel_software_dev() -> graphviz.Digraph:
        """
        Creates a directed graph representing the software development process
        and its dependencies using graphviz.

        Returns:
            graphviz.Digraph: A graphviz Digraph object representing the graph.
        """
        # Create a directed graph object
        dot = graphviz.Digraph(comment='Software Development Process', format='png')  # You can change format if needed

        # Add nodes for each stage
        dot.node('Requirements', 'Requirements\nSpecification')
        dot.node('Design', 'UI/UX Design')
        dot.node('Backend', 'Backend\nDevelopment')
        dot.node('Frontend', 'Frontend\nDevelopment')
        dot.node('Integration', 'Integration\n&\nTesting')
        dot.node('Deployment', 'Deployment\n&\nRelease')

        # Add edges to represent dependencies
        dot.edge('Requirements', 'Design')
        dot.edge('Requirements', 'Backend')
        dot.edge('Design', 'Frontend')
        dot.edge('Backend', 'Integration')
        dot.edge('Frontend', 'Integration')
        dot.edge('Integration', 'Deployment')

        # You can customize the graph appearance (optional)
        dot.attr('node', shape='box', style='rounded', fontname='Arial', fontsize='12')
        dot.attr('edge', arrowtail='empty', arrowhead='normal', color='darkgreen')
        dot.attr(size='8.5,11')  # Set the size of the output graph (optional)

        return dot

@dataclass
class WidgetContainer:
    name: str
    container: sim.Container

class WidgetContainerGenerator:

    def __init__(self, env: sim.Environment, name: str, concept_dist: np.array, n: int = 10) -> "WidgetContainerGenerator":
        self.env: sim.Environment = env
        self.name: str = name
        self.n: int = n
        self.concept_dist: np.array = concept_dist
        self.widget_containers: List[WidgetContainer] = self.generate_containers()

    def _generate_container(self, container_id: int) -> WidgetContainer:
        volume: int = np.max([int(np.random.choice(self.concept_dist, 1)[0]), 1])
        name: str = f"{self.name}_{container_id}_{volume}"
        return WidgetContainer(name=name, container=sim.Container(self.env, init=volume, capacity=volume))

    def generate_containers(self) -> List[WidgetContainer]:
        containers: List[sim.Container] = [self._generate_container(i) for i in range(self.n)]
        return containers

class WidgetStore:

    def __init__(self, env: sim.Environment, name: str, capacity: Union[int, float] = np.inf) -> "WidgetStore":
        self.env = env
        self.name = name
        self.store: sim.Store = sim.Store(env, capacity=capacity)

    def source_load(self, generator: WidgetContainerGenerator) -> None:
        for container in generator.widget_containers:
            self.store.put(container)
            logger.info(f"Container {container.name} added to store {self.name} at time {self.env.now}")


@dataclass
class Allocation:
    step: int
    label: str
    n_people: int

    @staticmethod
    def people_from_label(label: str, allocation: List["Allocation"]) -> int:
        """
        Returns the number of people in the allocation with the given label.
        """
        for alloc in allocation:
            if alloc.label == label:
                return alloc.n_people
        raise ValueError(f"Label {label} not found in allocation.")

@dataclass
class Stage:
    step: int
    label: str
    resource: sim.Resource


class Pipeline:

    def __init__(
        self, 
        env: sim.Environment, 
        name: str, allocations: List[Allocation], 
        process_dist: np.array,
        pipeline_source: WidgetStore,
        pipeline_sink: WidgetStore
    ) -> "Pipeline":
        self.env = env
        self.name = name
        self.allocations = allocations
        self.process_dist = process_dist
        self.pipeline_source = pipeline_source
        self.pipeline_sink = pipeline_sink
        self.stages = self._gen_stages()
        self.container_data: Dict[str, Union[float, int]] = {
            "pipeline": [],
            "source": [],
            "sink": [],
            "container": [],
            "num_widgets": [],
            "stage": [],
            "start_time": [],
            "end_time": [],
        }

    def _gen_stages(self) -> List[Stage]:
        stages: List[Stage] = []
        for allocation in self.allocations:
            resource: sim.Resource = sim.Resource(self.env, capacity=allocation.n_people)
            stage: Stage = Stage(step=allocation.step, label=allocation.label, resource=resource)
            stages.append(stage)
        return stages
    
    @staticmethod
    def print_resource_stats(resource: sim.Resource) -> None:
        logger.info(f"""

        Resource capacity: {resource.capacity}
        Resource queue: {len(resource.queue)}
        Resource users: {len(resource.users)}

          """)


    def _process_widget(self, container: WidgetContainer, stage: Stage, process_id: int) -> sim.events.Process:
        process_name: str = f"process_{container.name}_{process_id}"
        logger.info(f"Widget {process_name} for container {container.name} (level = {container.container.level})started at stage {stage.label} at time {self.env.now}")
        processing_time: int = int(np.random.choice(self.process_dist, 1)[0])
        with stage.resource.request() as req:
            # logger.info(f"Requesting resource at stage {stage.label} at time {self.env.now}")
            yield req  # Pauses here until the resource is available
            logger.info(f"\tProcessing widget at stage {stage.label} at time {self.env.now} with processing time {processing_time}")
            Pipeline.print_resource_stats(stage.resource)
            yield self.env.timeout(processing_time)
            logger.info(f"\tWidget processing complete at stage {stage.label} at time {self.env.now}")
            logger.info(f"\tContainer {container.name} has {container.container.level} widgets left at stage {stage.label} at time {self.env.now}")
        
        if stage == self.stages[-1]:
            yield container.container.get(1)


    def _process_container(self, container: WidgetContainer, stage: Stage) -> sim.events.Process:
        start_time: float = self.env.now
        logger.info(f"Processing container {container.name} at stage {stage.label} at time {start_time}")

        # Create a list of processes for all widgets in the container
        widget_processes = [
            self.env.process(self._process_widget(container, stage, i))
            for i in range(container.container.level)
        ]

        # Wait for all widget processes to complete
        yield sim.events.AllOf(self.env, widget_processes)
        end_time: float = self.env.now
        logger.info(f"Container {container.name} processing complete at stage {stage.label} at time {end_time}")
        self.container_data["pipeline"].append(self.name)
        self.container_data["source"].append(self.pipeline_source.name)
        self.container_data["sink"].append(self.pipeline_sink.name)
        self.container_data["container"].append(container.name)
        self.container_data["num_widgets"].append(container.container.capacity)
        self.container_data["stage"].append(stage.label)
        self.container_data["start_time"].append(start_time)
        self.container_data["end_time"].append(end_time)


    def run(self) -> sim.events.Process:
        logger.info(f"""
Containers in source store {self.pipeline_source.name} at time {self.env.now}:
{[c.name for c in self.pipeline_source.store.items]}
Containers in sink store {self.pipeline_sink.name} at time {self.env.now}:
{[c.name for c in self.pipeline_sink.store.items]}              
              """)
        for _ in range(len(self.pipeline_source.store.items)):
            current_container: WidgetContainer = yield self.pipeline_source.store.get()
            logger.info(f"Container {current_container.name} is being processed at time {self.env.now}")
            for stage in self.stages:
                yield self.env.process(self._process_container(current_container, stage))
            logger.info(f"Container {current_container.name} is being sent to sink at time {self.env.now}")
            self.pipeline_sink.store.put(current_container)
            yield self.env.timeout(0)
        logger.info(f"Pipeline {self.name} is complete at time {self.env.now}")


    def get_container_data(self, sim_num: Optional[int] = None) -> pd.DataFrame:
        """
        Returns the container data collected during the simulation.

        Returns:
            pd.DataFrame: A DataFrame containing the container data.
        """
        data: pd.DataFrame = pd.DataFrame(self.container_data)
        data["process_time"] = data["end_time"] - data["start_time"]
        data["container_id"] = data["container"].apply(lambda x: int(x.split("_")[-2]))
        data["num_resources"] = data["stage"].apply(
            lambda stage: Allocation.people_from_label(stage, self.allocations)
        )
        if sim_num:
            data["simulation"] = sim_num

        return data

    def get_container_data_xr(self, sim_num: int = 0) -> xr.Dataset:
        """
        Returns the container data collected during the simulation as an xarray Dataset.
        Returns:
            xr.Dataset: An xarray Dataset containing the container data.
        """
        data: pd.DataFrame = self.get_container_data(sim_num)
        def to_data_array(v: str) -> xr.DataArray:
            tmp_df: pd.DataFrame = data.set_index(["container_id", "stage"])[v]
            xarr: xr. DataArray = xr.DataArray(tmp_df.unstack())
            xarr = xarr.expand_dims({"pipeline": [self.name], "simulation": [sim_num]})
            return xarr

        data_arrays: Dict[str, xr.DataArray] = {
            "num_widgets": to_data_array("num_widgets"),
            "process_time": to_data_array("process_time"),
            "start_time": to_data_array("start_time"),
            "end_time": to_data_array("end_time"),
            "num_resources": to_data_array("num_resources"),
        }
        ds: xr.Dataset = xr.Dataset(data_arrays)
        ds.attrs = {"source": self.pipeline_source.name, "sink": self.pipeline_sink.name}
        return ds


    @staticmethod
    def simulations(
        n_sim: int, 
        n_containers: int,
        allocations: List[Allocation], 
        process_dist: np.array,
        concept_dist: np.array,
        pipeline_name: str = "allocation_0",
        pipeline_source_name: str = "source", 
        pipeline_sink_name: str = "sink",
        generator_name: str = "wgen",
    ) -> xr.Dataset:
        """
        Runs multiple simulations of the pipeline and returns the results as an xarray Dataset.

        Args:
            n (int): The number of simulations to run.
            allocations (List[Allocation]): A list of allocations for each stage.
            process_dist (np.array): The distribution of processing times.
            pipeline_source (WidgetStore): The source store for the pipeline.
            pipeline_sink (WidgetStore): The sink store for the pipeline.

        Returns:
            xr.Dataset: An xarray Dataset containing the results of the simulations.
        """

        ds_list: List[xr.Dataset] = []
        for i in range(n_sim):
            logger.info(f"Running simulation {i+1}/{n_sim}")
            env_i: sim.Environment = sim.Environment()
            wgen_i: WidgetContainerGenerator = WidgetContainerGenerator(
                env=env_i, 
                name=f"{generator_name}_{i}", 
                concept_dist=concept_dist,
                n=n_containers
            )
            source_i: WidgetStore = WidgetStore(env=env_i, name=f"{pipeline_source_name}_{i}")
            source_i.source_load(wgen_i)
            sink_i: WidgetStore = WidgetStore(env=env_i, name=f"{pipeline_sink_name}_{i}")
            pipeline: Pipeline = Pipeline(
                env_i, 
                pipeline_name,
                allocations, 
                process_dist, 
                source_i, 
                sink_i
            )
            env_i.process(pipeline.run())
            env_i.run()
            ds_list.append(pipeline.get_container_data_xr(i))
        
        return xr.concat(ds_list, dim="simulation")


    @staticmethod
    def simulations_across_allocations(
        n_sim: int, 
        n_containers: int,
        allocations: Dict[str, List[Allocation]], 
        process_dist: np.array,
        concept_dist: np.array,
        pipeline_source_name: str = "source", 
        pipeline_sink_name: str = "sink",
        generator_name: str = "wgen",
    ) -> xr.Dataset:

        for allocation_name, allocation_list in allocations.items():
            logger.info(f"Running simulations for allocation {allocation_name}")
            ds: xr.Dataset = Pipeline.simulations(
                n_sim=n_sim,
                n_containers=n_containers,
                allocations=allocation_list,
                process_dist=process_dist,
                concept_dist=concept_dist,
                pipeline_name=allocation_name,
                pipeline_source_name=pipeline_source_name,
                pipeline_sink_name=pipeline_sink_name,
                generator_name=generator_name
            )
            if allocation_name == list(allocations.keys())[0]:
                ds_allocation: xr.Dataset = ds
            else:
                ds_allocation = xr.concat([ds_allocation, ds], dim="pipeline")
        return ds_allocation

    
    @staticmethod
    def comparison_plot(
        ds: xr.Dataset, 
        measure: str = "process_time", 
        comparison_var: str = "pipeline",
        observation_var: str = "simulation",
        group_var: Optional[str] = None,
        transform: Callable[[xr.DataArray], xr.DataArray] = lambda x: x.mean(
            dim=["stage", "container_id"]
        ),
        figsize: Tuple[int, int] = (10, 6),
        ttl_in: Optional[str] = None
    ) -> plt.Figure:
        """
        Creates a comparison plot for the specified variable in the dataset.

        Args:
            ds (xr.Dataset): The xarray Dataset containing the simulation results.
            measure (str): The variable to plot. Default is "process_time".
            comparison_var (str): The variable determining groups upon which comparison is made (across groups)
            observation_var (str): The variable determining values in each distribution (within groups)
            figsize (Tuple[int, int]): The size of the figure. Default is (10, 6).
        """

        data: xr.DataArray = transform(ds[measure].groupby([comparison_var, observation_var]))
        plot_data: pd.DataFrame = data.to_dataframe().reset_index()

        fig, ax = plt.subplots(figsize=figsize)
        compare_plot: plt.Figure = sb.violinplot(
            data=plot_data,
            x=comparison_var,
            y=measure,
            hue=group_var,
            ax=ax
        )
        ttl: str = f"{measure} by {comparison_var}" if ttl_in is None else ttl_in
        ax.set_title(ttl)
        plt.show()