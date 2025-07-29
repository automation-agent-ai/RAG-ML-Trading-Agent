#!/usr/bin/env python3
"""
knowledge_graph_api.py - FastAPI endpoints for Knowledge Graph operations

Provides REST API endpoints for:
- Graph data and metadata
- Setup traversal and similarity analysis
- Reasoning path discovery
- Subgraph extraction
- Visualization data
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# FastAPI imports
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Local imports
from ml.knowledge_graph_builder import KnowledgeGraphDataPrep, KnowledgeGraphBuilder
from ml.knowledge_graph_traversal import (
    KnowledgeGraphTraversal, 
    TraversalResult, 
    SimilarSetup, 
    ReasoningPath,
    create_traversal_json
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
kg_router = APIRouter(prefix="/api/kg", tags=["Knowledge Graph"])

# Global variables
kg_traversal = None
kg_initialized = False
kg_stats = {}

# Pydantic models
class GraphStatsResponse(BaseModel):
    total_nodes: int
    total_edges: int
    feature_nodes: int
    reasoning_nodes: int
    setup_nodes: int
    graph_density: float
    last_updated: str
    initialization_status: str

class TraversalRequest(BaseModel):
    setup_id: str
    analysis_depth: str = Field(default="medium", pattern="^(shallow|medium|deep)$")
    include_similar: bool = True
    include_reasoning: bool = True
    max_similar: int = Field(default=10, ge=1, le=50)

class SimilarityRequest(BaseModel):
    setup_id: str
    top_k: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    include_reasoning: bool = True

class ReasoningPathRequest(BaseModel):
    setup_id: str
    path_types: List[str] = Field(default=["feature_chain", "reasoning_chain", "causal_chain"])
    max_path_length: int = Field(default=5, ge=2, le=10)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

class SubgraphRequest(BaseModel):
    setup_id: str
    max_hops: int = Field(default=3, ge=1, le=5)
    include_similar: bool = True
    include_reasoning: bool = True

class GraphInitializationRequest(BaseModel):
    force_rebuild: bool = False
    save_to_disk: bool = True


# Initialize knowledge graph
async def initialize_knowledge_graph():
    """Initialize the knowledge graph in the background"""
    global kg_traversal, kg_initialized, kg_stats
    
    logger.info("🔧 Initializing Knowledge Graph...")
    
    try:
        # Try to load existing graph
        graph_path = "../analysis_ml_ensemble/knowledge_graph.pkl"
        
        if os.path.exists(graph_path):
            logger.info("📂 Loading existing knowledge graph...")
            kg_traversal = KnowledgeGraphTraversal(graph_path=graph_path)
        else:
            logger.info("🏗️ Building new knowledge graph...")
            data_prep = KnowledgeGraphDataPrep()
            kg_builder = KnowledgeGraphBuilder(data_prep)
            graph = kg_builder.build_complete_graph()
            kg_builder.save_graph(graph_path)
            kg_traversal = KnowledgeGraphTraversal(graph=graph)
        
        # Calculate statistics
        kg_stats = {
            'total_nodes': kg_traversal.graph.number_of_nodes(),
            'total_edges': kg_traversal.graph.number_of_edges(),
            'feature_nodes': len(kg_traversal.feature_nodes),
            'reasoning_nodes': len(kg_traversal.reasoning_nodes),
            'setup_nodes': len(kg_traversal.setup_nodes),
            'graph_density': kg_traversal.graph.number_of_edges() / max(kg_traversal.graph.number_of_nodes() * (kg_traversal.graph.number_of_nodes() - 1), 1),
            'last_updated': datetime.now().isoformat(),
            'initialization_status': 'completed'
        }
        
        kg_initialized = True
        logger.info("✅ Knowledge Graph initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Error initializing Knowledge Graph: {e}")
        kg_stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'feature_nodes': 0,
            'reasoning_nodes': 0,
            'setup_nodes': 0,
            'graph_density': 0.0,
            'last_updated': datetime.now().isoformat(),
            'initialization_status': f'failed: {str(e)}'
        }
        kg_initialized = False


# API Endpoints

@kg_router.get("/status", response_model=GraphStatsResponse)
async def get_graph_status():
    """Get current status and statistics of the knowledge graph"""
    if not kg_initialized:
        return GraphStatsResponse(
            total_nodes=0,
            total_edges=0,
            feature_nodes=0,
            reasoning_nodes=0,
            setup_nodes=0,
            graph_density=0.0,
            last_updated=datetime.now().isoformat(),
            initialization_status="not_initialized"
        )
    
    return GraphStatsResponse(**kg_stats)


@kg_router.post("/initialize")
async def initialize_graph(
    request: GraphInitializationRequest,
    background_tasks: BackgroundTasks
):
    """Initialize or rebuild the knowledge graph"""
    global kg_initialized, kg_stats
    
    if request.force_rebuild or not kg_initialized:
        # Reset status
        kg_initialized = False
        kg_stats['initialization_status'] = 'initializing'
        
        # Start initialization in background
        background_tasks.add_task(initialize_knowledge_graph)
        
        return {"message": "Knowledge graph initialization started", "status": "initializing"}
    
    return {"message": "Knowledge graph already initialized", "status": "completed"}


@kg_router.post("/traverse")
async def traverse_setup(request: TraversalRequest):
    """Perform complete traversal analysis for a setup"""
    if not kg_initialized:
        raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
    
    try:
        # Validate setup exists
        if request.setup_id not in kg_traversal.setup_nodes:
            raise HTTPException(status_code=404, detail=f"Setup {request.setup_id} not found")
        
        # Perform traversal
        result = kg_traversal.complete_traversal(
            query_setup_id=request.setup_id,
            analysis_depth=request.analysis_depth
        )
        
        # Filter results based on request parameters
        if not request.include_similar:
            result.similar_setups = []
        elif len(result.similar_setups) > request.max_similar:
            result.similar_setups = result.similar_setups[:request.max_similar]
        
        if not request.include_reasoning:
            result.reasoning_paths = []
        
        return create_traversal_json(result)
        
    except Exception as e:
        logger.error(f"Error during traversal: {e}")
        raise HTTPException(status_code=500, detail=f"Traversal failed: {str(e)}")


@kg_router.post("/similarity")
async def find_similar_setups(request: SimilarityRequest):
    """Find setups similar to the given setup"""
    if not kg_initialized:
        raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
    
    try:
        # Validate setup exists
        if request.setup_id not in kg_traversal.setup_nodes:
            raise HTTPException(status_code=404, detail=f"Setup {request.setup_id} not found")
        
        # Find similar setups
        similar_setups = kg_traversal.find_similar_setups(
            query_setup_id=request.setup_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            include_reasoning=request.include_reasoning
        )
        
        return {
            "query_setup_id": request.setup_id,
            "similar_setups": [
                {
                    "setup_id": setup.setup_id,
                    "similarity_score": setup.similarity_score,
                    "common_features": setup.common_features,
                    "shared_reasoning": setup.shared_reasoning,
                    "outcome": setup.outcome,
                    "key_differences": setup.key_differences,
                    "explanation": setup.explanation
                } for setup in similar_setups
            ],
            "total_found": len(similar_setups),
            "request_parameters": request.dict()
        }
        
    except Exception as e:
        logger.error(f"Error finding similar setups: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")


@kg_router.post("/reasoning")
async def discover_reasoning_paths(request: ReasoningPathRequest):
    """Discover reasoning paths for a setup"""
    if not kg_initialized:
        raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
    
    try:
        # Validate setup exists
        if request.setup_id not in kg_traversal.setup_nodes:
            raise HTTPException(status_code=404, detail=f"Setup {request.setup_id} not found")
        
        # Discover reasoning paths
        reasoning_paths = kg_traversal.discover_reasoning_paths(
            query_setup_id=request.setup_id,
            path_types=request.path_types,
            max_path_length=request.max_path_length,
            min_confidence=request.min_confidence
        )
        
        return {
            "query_setup_id": request.setup_id,
            "reasoning_paths": [
                {
                    "path_id": path.path_id,
                    "nodes": path.nodes,
                    "path_type": path.path_type,
                    "confidence": path.confidence,
                    "setup_examples": path.setup_examples,
                    "pattern_strength": path.pattern_strength,
                    "explanation": path.explanation
                } for path in reasoning_paths
            ],
            "total_found": len(reasoning_paths),
            "request_parameters": request.dict()
        }
        
    except Exception as e:
        logger.error(f"Error discovering reasoning paths: {e}")
        raise HTTPException(status_code=500, detail=f"Reasoning path discovery failed: {str(e)}")


@kg_router.post("/subgraph")
async def extract_subgraph(request: SubgraphRequest):
    """Extract a subgraph centered around a setup"""
    if not kg_initialized:
        raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
    
    try:
        # Validate setup exists
        if request.setup_id not in kg_traversal.setup_nodes:
            raise HTTPException(status_code=404, detail=f"Setup {request.setup_id} not found")
        
        # Extract subgraph
        subgraph_data = kg_traversal.extract_setup_subgraph(
            query_setup_id=request.setup_id,
            max_hops=request.max_hops,
            include_similar=request.include_similar,
            include_reasoning=request.include_reasoning
        )
        
        return subgraph_data
        
    except Exception as e:
        logger.error(f"Error extracting subgraph: {e}")
        raise HTTPException(status_code=500, detail=f"Subgraph extraction failed: {str(e)}")


@kg_router.get("/nodes")
async def get_all_nodes(
    node_type: Optional[str] = Query(None, description="Filter by node type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of nodes to return")
):
    """Get all nodes in the knowledge graph"""
    if not kg_initialized:
        raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
    
    try:
        nodes = []
        node_count = 0
        
        for node_id, node_data in kg_traversal.graph.nodes(data=True):
            if node_count >= limit:
                break
                
            if node_type is None or node_data.get('node_type') == node_type:
                nodes.append({
                    "node_id": node_id,
                    "node_type": node_data.get('node_type', 'unknown'),
                    "data": node_data
                })
                node_count += 1
        
        return {
            "nodes": nodes,
            "total_returned": len(nodes),
            "filters": {"node_type": node_type, "limit": limit}
        }
        
    except Exception as e:
        logger.error(f"Error getting nodes: {e}")
        raise HTTPException(status_code=500, detail=f"Node retrieval failed: {str(e)}")


@kg_router.get("/edges")
async def get_all_edges(
    edge_type: Optional[str] = Query(None, description="Filter by edge type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of edges to return")
):
    """Get all edges in the knowledge graph"""
    if not kg_initialized:
        raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
    
    try:
        edges = []
        edge_count = 0
        
        for source, target, edge_data in kg_traversal.graph.edges(data=True):
            if edge_count >= limit:
                break
                
            if edge_type is None or edge_data.get('edge_type') == edge_type:
                edges.append({
                    "source": source,
                    "target": target,
                    "edge_type": edge_data.get('edge_type', 'unknown'),
                    "weight": edge_data.get('weight', 1.0),
                    "confidence": edge_data.get('confidence', 0.5),
                    "metadata": edge_data.get('metadata', {})
                })
                edge_count += 1
        
        return {
            "edges": edges,
            "total_returned": len(edges),
            "filters": {"edge_type": edge_type, "limit": limit}
        }
        
    except Exception as e:
        logger.error(f"Error getting edges: {e}")
        raise HTTPException(status_code=500, detail=f"Edge retrieval failed: {str(e)}")


@kg_router.get("/setup/{setup_id}")
async def get_setup_details(setup_id: str):
    """Get detailed information about a specific setup"""
    if not kg_initialized:
        raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
    
    try:
        # Validate setup exists
        if setup_id not in kg_traversal.setup_nodes:
            raise HTTPException(status_code=404, detail=f"Setup {setup_id} not found")
        
        # Get setup details
        setup_data = kg_traversal.setup_nodes[setup_id]
        
        # Get connected features
        connected_features = []
        if setup_id in kg_traversal.graph:
            for neighbor in kg_traversal.graph.neighbors(setup_id):
                node_data = kg_traversal.graph.nodes[neighbor]
                if node_data.get('node_type') == 'feature':
                    connected_features.append({
                        "feature_id": neighbor,
                        "feature_name": node_data.get('feature_name', 'unknown'),
                        "feature_type": node_data.get('feature_type', 'unknown'),
                        "feature_category": node_data.get('feature_category', 'unknown')
                    })
        
        # Get reasoning nodes
        reasoning_nodes = []
        for node_id, node_data in kg_traversal.reasoning_nodes.items():
            if setup_id in node_id:
                reasoning_nodes.append({
                    "reasoning_id": node_id,
                    "agent_type": node_data.get('agent_type', 'unknown'),
                    "reasoning_step": node_data.get('reasoning_step', 'unknown'),
                    "reasoning_text": node_data.get('reasoning_text', ''),
                    "confidence": node_data.get('confidence', 0.0)
                })
        
        return {
            "setup_id": setup_id,
            "setup_data": setup_data,
            "connected_features": connected_features,
            "reasoning_nodes": reasoning_nodes,
            "total_connected_features": len(connected_features),
            "total_reasoning_nodes": len(reasoning_nodes)
        }
        
    except Exception as e:
        logger.error(f"Error getting setup details: {e}")
        raise HTTPException(status_code=500, detail=f"Setup details retrieval failed: {str(e)}")


@kg_router.get("/visualization/overview")
async def get_visualization_overview():
    """Get overview data for knowledge graph visualization"""
    if not kg_initialized:
        raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
    
    try:
        # Calculate node statistics by type
        node_stats = {}
        for node_id, node_data in kg_traversal.graph.nodes(data=True):
            node_type = node_data.get('node_type', 'unknown')
            node_stats[node_type] = node_stats.get(node_type, 0) + 1
        
        # Calculate edge statistics by type
        edge_stats = {}
        for source, target, edge_data in kg_traversal.graph.edges(data=True):
            edge_type = edge_data.get('edge_type', 'unknown')
            edge_stats[edge_type] = edge_stats.get(edge_type, 0) + 1
        
        # Get feature type distribution
        feature_type_stats = {}
        for node_id, node_data in kg_traversal.feature_nodes.items():
            feature_type = node_data.get('feature_type', 'unknown')
            feature_type_stats[feature_type] = feature_type_stats.get(feature_type, 0) + 1
        
        # Get outcome distribution
        outcome_stats = {"successful": 0, "failed": 0}
        for setup_id, setup_data in kg_traversal.setup_nodes.items():
            if setup_data.get('outperformed', False):
                outcome_stats["successful"] += 1
            else:
                outcome_stats["failed"] += 1
        
        return {
            "node_statistics": node_stats,
            "edge_statistics": edge_stats,
            "feature_type_distribution": feature_type_stats,
            "outcome_distribution": outcome_stats,
            "total_nodes": kg_traversal.graph.number_of_nodes(),
            "total_edges": kg_traversal.graph.number_of_edges(),
            "graph_density": kg_traversal.graph.number_of_edges() / max(kg_traversal.graph.number_of_nodes() * (kg_traversal.graph.number_of_nodes() - 1), 1)
        }
        
    except Exception as e:
        logger.error(f"Error getting visualization overview: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization overview failed: {str(e)}")


@kg_router.get("/visualization/graph")
async def get_graph_visualization_data(
    max_nodes: int = Query(100, ge=10, le=500, description="Maximum number of nodes to return"),
    include_features: bool = Query(True, description="Include feature nodes"),
    include_reasoning: bool = Query(False, description="Include reasoning nodes"),
    include_setups: bool = Query(True, description="Include setup nodes")
):
    """Get graph data formatted for D3.js visualization"""
    if not kg_initialized:
        raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
    
    try:
        nodes = []
        edges = []
        node_count = 0
        
        # Add nodes based on filters
        for node_id, node_data in kg_traversal.graph.nodes(data=True):
            if node_count >= max_nodes:
                break
                
            node_type = node_data.get('node_type', 'unknown')
            
            should_include = (
                (include_features and node_type == 'feature') or
                (include_reasoning and node_type == 'reasoning') or
                (include_setups and node_type == 'setup') or
                (node_type == 'outcome')
            )
            
            if should_include:
                # Create node for visualization
                viz_node = {
                    "id": node_id,
                    "type": node_type,
                    "label": node_data.get('feature_name', node_id),
                    "group": node_data.get('feature_type', node_type),
                    "size": 10,  # Default size
                    "metadata": node_data
                }
                
                # Adjust size based on node type
                if node_type == 'setup':
                    viz_node["size"] = 15
                elif node_type == 'outcome':
                    viz_node["size"] = 20
                elif node_type == 'feature':
                    viz_node["size"] = 12
                
                nodes.append(viz_node)
                node_count += 1
        
        # Add edges between included nodes
        included_node_ids = {node["id"] for node in nodes}
        
        for source, target, edge_data in kg_traversal.graph.edges(data=True):
            if source in included_node_ids and target in included_node_ids:
                edges.append({
                    "source": source,
                    "target": target,
                    "type": edge_data.get('edge_type', 'unknown'),
                    "weight": edge_data.get('weight', 1.0),
                    "confidence": edge_data.get('confidence', 0.5),
                    "metadata": edge_data.get('metadata', {})
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "visualization_parameters": {
                "max_nodes": max_nodes,
                "include_features": include_features,
                "include_reasoning": include_reasoning,
                "include_setups": include_setups
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting graph visualization data: {e}")
        raise HTTPException(status_code=500, detail=f"Graph visualization data failed: {str(e)}")


# Initialize the knowledge graph when the module is imported
# This will be called when the main app starts
def init_kg_on_startup():
    """Initialize knowledge graph on startup"""
    import threading
    
    def init_thread():
        asyncio.run(initialize_knowledge_graph())
    
    # Start initialization in a separate thread
    init_thread = threading.Thread(target=init_thread)
    init_thread.daemon = True
    init_thread.start()


# Export the router for use in main app
__all__ = ['kg_router', 'init_kg_on_startup'] 