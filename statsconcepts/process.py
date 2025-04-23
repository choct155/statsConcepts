import matplotlib.pyplot as plt
import seaborn as sb
import xarray as xr
import pymc as pm
import numpy as np
import graphviz
import arviz as az
import pytensor as pt
from typing import Optional, Tuple, Set, List, Dict
from dataclasses import dataclass


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
