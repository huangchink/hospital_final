"""Small pygame input/focus helpers shared across the pygame-driven apps/flows."""
import time

import pygame


def restore_event_filter():
    try:
        pygame.event.set_allowed(None)
        pygame.event.clear()
        pygame.event.pump()
    except Exception:
        pass


def ensure_pygame_focus(timeout=2.0):
    t0 = time.time()
    while not pygame.key.get_focused():
        pygame.event.pump()
        if time.time() - t0 > timeout:
            break
        time.sleep(0.02)
