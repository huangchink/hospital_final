"""Sol SDK asyncio worker runner (run_sol_worker) - extracted from VA_center_opt.py."""
import asyncio
import traceback


def run_sol_worker(connector, on_connect, on_fail):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(connector.run_session(on_connect, on_fail))
    except Exception as e:
        print(f"[Sol Worker] Unexpected error: {e}")
        traceback.print_exc()
        if on_fail:
            on_fail(f"Unexpected error: {e}")
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for t in pending: t.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as cleanup_error:
            print(f"[Sol Worker] Cleanup error: {cleanup_error}")
        finally:
            try:
                loop.close()
            except Exception:
                pass
