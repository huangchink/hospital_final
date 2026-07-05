"""Windows keyboard-layout management.

Switches the active keyboard layout to English (US) so keystroke controls (q, SPACE,
...) fire regardless of the user's IME state, and restores the original layout on exit
- with crash recovery via a small backup file next to the app.

Shared by the VA/VF suite (VA_center_opt) and the webcam calibration tool
(calibration). Extracted verbatim from VA_center_opt.py; stdlib-only (ctypes + the
stdlib APP_DIR) so it stays import-light for VA_center_opt's multiprocessing 'spawn'
children.
"""
import ctypes

from ntuh.common.app_env import APP_DIR


# [FIX] Windows keyboard layout management - switch to English to ensure
# keystroke-based controls (q, SPACE, etc.) work regardless of IME state.
# Persists original layout to file for crash recovery.
class KeyboardLayoutManager:
    """Cache current keyboard layout, switch to English, restore on exit.
    Saves original layout to a file so it can be recovered after a native crash."""
    EN_US_LAYOUT = 0x0409
    _RECOVERY_FILE = APP_DIR / ".kb_layout_backup"

    def __init__(self):
        self._original_layout = None
        self._switched = False
        # On startup, check if a previous crash left the keyboard in English
        self._recover_from_crash()

    def _recover_from_crash(self):
        """If a previous run crashed without restoring, restore now."""
        try:
            if self._RECOVERY_FILE.exists():
                layout_hex = self._RECOVERY_FILE.read_text().strip()
                if layout_hex:
                    layout = int(layout_hex, 16)
                    user32 = ctypes.windll.user32
                    current_lang = user32.GetKeyboardLayout(0) & 0xFFFF
                    if current_lang == self.EN_US_LAYOUT and (layout & 0xFFFF) != self.EN_US_LAYOUT:
                        # Previous crash left us in English, restore
                        user32.ActivateKeyboardLayout(layout, 0)
                        user32.PostMessageW(0xFFFF, 0x0050, 0, layout)
                        print(f"[Keyboard] Recovered from previous crash - restored layout 0x{layout:08X}")
                self._RECOVERY_FILE.unlink(missing_ok=True)
        except Exception as e:
            print(f"[Keyboard] Crash recovery check failed: {e}")

    def switch_to_english(self):
        """Cache current layout and switch to English (US)."""
        try:
            user32 = ctypes.windll.user32
            self._original_layout = user32.GetKeyboardLayout(0)
            current_lang = self._original_layout & 0xFFFF
            if current_lang != self.EN_US_LAYOUT:
                # Save original layout to file for crash recovery
                try:
                    self._RECOVERY_FILE.write_text(f"0x{self._original_layout:08X}")
                except Exception:
                    pass
                # Load and activate English (US) layout
                hkl = user32.LoadKeyboardLayoutW(f"{self.EN_US_LAYOUT:08X}", 0x01)  # KLF_ACTIVATE
                if hkl:
                    user32.PostMessageW(0xFFFF, 0x0050, 0, hkl)
                    self._switched = True
                    print(f"[Keyboard] Switched to English (US) from layout 0x{self._original_layout:08X}")
                else:
                    print("[Keyboard] Failed to load English layout")
                    self._RECOVERY_FILE.unlink(missing_ok=True)
            else:
                print("[Keyboard] Already using English layout")
        except Exception as e:
            print(f"[Keyboard] Error switching layout: {e}")

    def restore(self):
        """Restore the original keyboard layout."""
        if not self._switched or self._original_layout is None:
            return
        try:
            user32 = ctypes.windll.user32
            user32.ActivateKeyboardLayout(self._original_layout, 0)
            user32.PostMessageW(0xFFFF, 0x0050, 0, self._original_layout)
            print(f"[Keyboard] Restored original layout 0x{self._original_layout:08X}")
            self._switched = False
        except Exception as e:
            print(f"[Keyboard] Error restoring layout: {e}")
        finally:
            # Remove recovery file - we restored successfully
            try:
                self._RECOVERY_FILE.unlink(missing_ok=True)
            except Exception:
                pass
