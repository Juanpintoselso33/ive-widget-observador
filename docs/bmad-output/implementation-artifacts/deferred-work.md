# Deferred Work

## Spec: balotaje-2024-iframe-embed (2026-05-18)

### HTML embed UX — dos snippets pueden confundir a editores no técnicos
**Finding:** El archivo `docs/embed/ive-widget-embed.html` presenta dos bloques alternativos (estático + JS). Editores no técnicos podrían pegar ambos o elegir el incorrecto.
**Propuesta futura:** Consolidar en un único snippet con un comentario claro al inicio: "Copiar solo este bloque". Alternativamente, separar en dos archivos distintos.

### JS height resizer no se adapta a contenido dinámico de Streamlit
**Finding:** El resizer JS calcula altura por `offsetWidth < 480` (ancho del contenedor), no por la altura real del iframe. Si Streamlit renderiza más contenido del esperado, el iframe puede quedar cortado.
**Propuesta futura:** Evaluar postMessage desde Streamlit para enviar la altura real del documento, o investigar si Streamlit Cloud expone algún mecanismo de resize communication.

### Riesgo de colisión de `id="ive-widget"` si el snippet se pega dos veces
**Finding:** El snippet JS variant usa `document.getElementById('ive-widget')`, que fallaría silenciosamente si hay dos iframes en la misma página.
**Propuesta futura:** Usar un selector más robusto o agregar nota explícita en los comentarios del HTML: "Solo un embed por página".
