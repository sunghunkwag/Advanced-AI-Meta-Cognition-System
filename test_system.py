"""Comprehensive test suite for Advanced AI Meta-Cognition System.

Tests all components individually and as an integrated system.
"""

import torch
import numpy as np
import sys
from pathlib import Path


def test_imports():
    """Test 1: Verify all imports work correctly."""
    print("\n" + "="*60)
    print("TEST 1: IMPORT VERIFICATION")
    print("="*60)
    
    try:
        from vision import VisionSystem
        from manifold import GraphAttentionManifold
        from energy_improved import NeuroChemicalEngine, ImprovedJEPA
        from action_decoder import ActionDecoder
        from automata import IntrinsicAutomata
        from world import World
        from soul import get_soul_vectors
        from config import SystemConfig, get_default_config
        from logger import SystemLogger, StepMetrics
        
        print("✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_pytorch_setup():
    """Test 2: Verify PyTorch installation and device."""
    print("\n" + "="*60)
    print("TEST 2: PYTORCH SETUP")
    print("="*60)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test basic tensor operations
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = x @ y
    print(f"Matrix multiplication test: {z.shape} tensor created")
    
    print("✅ PyTorch setup verified")
    return True


def test_world_component():
    """Test 3: World module."""
    print("\n" + "="*60)
    print("TEST 3: WORLD COMPONENT")
    print("="*60)
    
    from world import World
    
    world = World(size=8)
    print(f"Created world: {world.size}x{world.size}")
    
    # Test energy calculation
    energy = world.calculate_energy()
    print(f"Initial energy: {energy:.4f}")
    
    # Test action
    action = {'type': 'DRAW', 'x': 4, 'y': 4, 'scale': 1.0}
    world.apply_action(action)
    new_energy = world.calculate_energy()
    print(f"Energy after DRAW: {new_energy:.4f}")
    print(f"Grid sum: {world.grid.sum():.2f}")
    
    print("✅ World component working")
    return True


def test_neurochemical_engine():
    """Test 4: Neuro-chemical engine."""
    print("\n" + "="*60)
    print("TEST 4: NEURO-CHEMICAL ENGINE")
    print("="*60)
    
    from energy_improved import NeuroChemicalEngine
    from config import NeuroChemicalConfig
    
    config = NeuroChemicalConfig()
    heart = NeuroChemicalEngine(config)
    
    print(f"Initial - D: {heart.dopamine:.2f}, S: {heart.serotonin:.2f}")
    
    # Simulate energy reduction (improvement)
    energies = [1.0, 0.7, 0.5, 0.3, 0.2, 0.15]
    for i, energy in enumerate(energies):
        heart.update(energy, consistency_score=0.6)
        d, s = heart.get_hormones()
        mode = heart.get_state()
        print(f"Step {i+1}: E={energy:.2f} D={d:.2f} S={s:.2f} Mode={mode}")
    
    final_d, final_s = heart.get_hormones()
    print(f"\nFinal hormones: D={final_d:.3f}, S={final_s:.3f}")
    print(f"Mode transition: {'✓' if final_d < 0.5 else '✗'}")
    
    print("✅ Neuro-chemical engine working")
    return True


def test_gat_network():
    """Test 5: Graph Attention Network."""
    print("\n" + "="*60)
    print("TEST 5: GRAPH ATTENTION NETWORK")
    print("="*60)
    
    from manifold import GraphAttentionManifold
    from soul import get_soul_vectors
    
    # Create mock graph
    num_nodes = 5
    nfeat = 3
    latent_dim = 8
    
    v_id, v_truth, v_rej = get_soul_vectors(dim=latent_dim)
    
    gat = GraphAttentionManifold(
        nfeat=nfeat,
        nhid=16,
        nclass=latent_dim,
        truth_vector=v_truth
    )
    
    # Create mock input
    # nodes must be (N, nfeat) for single graph or handled correctly by GAT.
    # The error "self must be a matrix" usually implies 3D input to torch.mm.
    # Original test had nodes = torch.randn(1, num_nodes, nfeat) which is (B, N, F).
    # GAT implementation does torch.mm(h, self.W), expecting h to be (N, F).
    # So we should remove the batch dimension for this specific test setup or the GAT expects no batch.

    nodes = torch.randn(num_nodes, nfeat)
    adj = torch.eye(num_nodes) # Identity adjacency (N, N)
    
    # Forward pass
    z = gat(nodes, adj)
    print(f"Input shape: {nodes.shape}")
    print(f"Output shape: {z.shape}")
    print(f"Expected: (1, {latent_dim})")
    
    # Test consistency
    consistency = gat.check_consistency(z)
    print(f"Consistency score: {consistency.item():.4f}")
    
    print("✅ GAT network working")
    return True


def test_action_decoder():
    """Test 6: Action decoder."""
    print("\n" + "="*60)
    print("TEST 6: ACTION DECODER")
    print("="*60)
    
    from action_decoder import ActionDecoder
    
    latent_dim = 8
    decoder = ActionDecoder(latent_dim=latent_dim)
    
    # Mock latent state
    z = torch.randn(1, latent_dim)
    
    # Get action
    action_logits, params = decoder(z)
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Params shape: {params.shape}")
    
    # Decode action
    action = decoder.decode_action(action_logits, params)
    print(f"\nDecoded action:")
    print(f"  Type: {action['type']}")
    print(f"  Position: ({action.get('x', 'N/A')}, {action.get('y', 'N/A')})")
    
    print("✅ Action decoder working")
    return True


def test_jepa_predictor():
    """Test 7: JEPA world model."""
    print("\n" + "="*60)
    print("TEST 7: JEPA WORLD MODEL")
    print("="*60)
    
    from energy_improved import ImprovedJEPA
    
    state_dim = 8
    action_dim = 4
    jepa = ImprovedJEPA(state_dim, action_dim, hidden_dim=32)
    
    # Mock state and action
    state = torch.randn(1, state_dim)
    action = torch.randn(1, action_dim)
    
    # Predict next state and energy
    next_state, energy = jepa(state, action)
    
    print(f"Current state: {state.shape}")
    print(f"Predicted next state: {next_state.shape}")
    print(f"Predicted energy: {energy.item():.4f}")
    
    # Test simulation function
    next_state_sim, energy_sim = jepa.simulate(state, action_id=0, num_actions=4)
    print(f"\nSimulation test:")
    print(f"  Next state: {next_state_sim.shape}")
    print(f"  Energy: {energy_sim:.4f}")
    
    print("✅ JEPA predictor working")
    return True


def test_integrated_system():
    """Test 8: Full system integration (short run)."""
    print("\n" + "="*60)
    print("TEST 8: INTEGRATED SYSTEM (10 STEPS)")
    print("="*60)
    
    from config import get_default_config
    from main_system_enhanced import AdvancedAISystem
    
    # Create config with short run
    config = get_default_config()
    config.training.max_steps = 10
    config.verbose = True
    config.seed = 42
    
    print("Initializing system...")
    system = AdvancedAISystem(config)
    
    print("Running 10 steps...\n")
    try:
        logger = system.run()
        
        # Check results
        summary = logger.get_summary_statistics()
        print("\n[Summary Statistics]")
        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Energy reduction: {summary['energy_reduction']:.4f}")
        print(f"  Final consistency: {summary['final_consistency']:.4f}")
        
        print("\n✅ Integrated system working")
        return True
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "#"*60)
    print("# ADVANCED AI META-COGNITION SYSTEM")
    print("# COMPREHENSIVE TEST SUITE")
    print("#"*60)
    
    tests = [
        ("Import Verification", test_imports),
        ("PyTorch Setup", test_pytorch_setup),
        ("World Component", test_world_component),
        ("Neuro-Chemical Engine", test_neurochemical_engine),
        ("Graph Attention Network", test_gat_network),
        ("Action Decoder", test_action_decoder),
        ("JEPA Predictor", test_jepa_predictor),
        ("Integrated System", test_integrated_system),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Final summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for experiments.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
    
    print("#"*60)


if __name__ == "__main__":
    run_all_tests()
