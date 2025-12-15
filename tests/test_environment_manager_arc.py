import numpy as np

from environment_manager_arc import ARCEnvironment, load_mini_arc_suite


def test_load_mini_arc_suite_produces_three_tasks():
    env = load_mini_arc_suite()
    assert isinstance(env, ARCEnvironment)
    assert len(env.tasks) == 3


def test_execute_action_improves_iou_on_simple_fill():
    env = load_mini_arc_suite()
    start_iou = env._iou(*env.get_state())
    new_iou = env.execute_action("FILL", {"color": 1})
    assert new_iou >= start_iou


def test_cycle_task_changes_target_grid():
    env = load_mini_arc_suite()
    _, first_target = env.get_state()
    env.cycle_task()
    _, second_target = env.get_state()
    assert not np.array_equal(first_target, second_target)
