from textual.binding import Binding

from kon.ui.app import Kon


def _binding_key_and_action(binding) -> tuple[str, str]:
    if isinstance(binding, Binding):
        return binding.key, binding.action
    key, action, *_ = binding
    return key, action


def test_thinking_and_permission_mode_keybindings():
    bindings = dict(_binding_key_and_action(binding) for binding in Kon.BINDINGS)

    assert bindings["ctrl+t"] == "toggle_thinking"
    assert bindings["ctrl+shift+t"] == "cycle_thinking_level"
    assert bindings["shift+tab"] == "cycle_permission_mode"
