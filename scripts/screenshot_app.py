"""
Captura screenshots de la app desplegada para revisar la estética.
Streamlit Cloud sirve la app dentro de un iframe; navegamos directo al src.
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

# Iframe interno (sin chrome de Streamlit Cloud)
URL = "https://juanpintoselso33-ive-widget-observador-app-poekzu.streamlit.app/~/+/"
OUT = Path(__file__).resolve().parent.parent / "screenshots"
OUT.mkdir(exist_ok=True)


async def wake_app(page):
    try:
        await page.get_by_role("button", name="Yes, get this app back up!").click(timeout=3000)
        await page.wait_for_timeout(3000)
    except Exception:
        pass


async def wait_streamlit_ready(page, timeout_ms=60_000):
    await page.wait_for_load_state("networkidle", timeout=timeout_ms)
    try:
        await page.wait_for_function(
            """() => {
                const sw = document.querySelector('[data-testid="stStatusWidget"]');
                return !sw || !sw.textContent.toLowerCase().includes('running');
            }""",
            timeout=timeout_ms,
        )
    except Exception:
        pass
    await page.wait_for_timeout(800)


async def shot(page, name):
    path = OUT / f"{name}.png"
    await page.screenshot(path=str(path), full_page=True)
    print(f"  -> {path.name}")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page = await ctx.new_page()

        print(f"Navegando a {URL}")
        await page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        await wake_app(page)
        await wait_streamlit_ready(page)
        await page.wait_for_timeout(2000)
        await shot(page, "01_initial_full")

        # Tabs
        tabs = page.locator('button[role="tab"]')
        count = await tabs.count()
        print(f"Tabs: {count}")
        tab_files = [
            "02_tab_religiosidad",
            "03_tab_balotaje",
            "04_tab_educacion",
            "05_tab_edad",
        ]
        for i, fname in enumerate(tab_files):
            if i < count:
                try:
                    await tabs.nth(i).scroll_into_view_if_needed()
                    await tabs.nth(i).click()
                    await page.wait_for_timeout(700)
                    await shot(page, fname)
                except Exception as e:
                    print(f"  !! tab {i} falló: {e}")

        # Expander metodología
        try:
            exp = page.locator('details summary').first
            await exp.scroll_into_view_if_needed()
            await exp.click()
            await page.wait_for_timeout(600)
            await shot(page, "06_metodologia_open")
        except Exception as e:
            print(f"  !! expander falló: {e}")

        # Volver arriba
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(500)

        # Cambiar a perfil "muy religioso + Lacalle"
        selects = page.locator('div[data-baseweb="select"]')
        scount = await selects.count()
        print(f"Selects: {scount}")
        try:
            # Religiosidad (índice 4 en col2)
            await selects.nth(4).click()
            await page.wait_for_timeout(400)
            await page.get_by_role("option", name="Mucho").click()
            await wait_streamlit_ready(page, 30_000)

            # Balotaje (índice 3 en col1)
            await selects.nth(3).click()
            await page.wait_for_timeout(400)
            await page.get_by_role("option", name="Lacalle (Coalición)").click()
            await wait_streamlit_ready(page, 30_000)

            await page.wait_for_timeout(800)
            await shot(page, "07_perfil_contra_full")
        except Exception as e:
            print(f"  !! perfil contra falló: {e}")

        # Dropdown educación abierto (4 opciones tras el colapso)
        try:
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(400)
            await selects.nth(2).click()
            await page.wait_for_timeout(600)
            await shot(page, "08_dropdown_educacion")
            await page.keyboard.press("Escape")
        except Exception as e:
            print(f"  !! dropdown educ falló: {e}")

        # Mobile
        await ctx.close()
        ctx_mobile = await browser.new_context(viewport={"width": 390, "height": 844})
        page_m = await ctx_mobile.new_page()
        await page_m.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        await wake_app(page_m)
        await wait_streamlit_ready(page_m)
        await page_m.wait_for_timeout(1500)
        await shot(page_m, "09_mobile_full")

        await browser.close()
        print("Listo.")


if __name__ == "__main__":
    asyncio.run(main())
