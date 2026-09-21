/* ===========================================================================
   Script de embebido del widget IVE — El Observador.

   Se usa igual que el de Flourish: un div con un atributo y este script al
   lado. No hay que poner altura.

     <div class="ive-embed" data-src="https://…/ive/"></div>
     <script src="https://…/ive/embed.js"></script>

   Para la versión de caja, agregarle `?resumen=1` al data-src.

   QUÉ RESUELVE. Un iframe de otro dominio no deja que la página que lo
   contiene mida su contenido, así que el alto hay que fijarlo a mano — y un
   alto fijo o recorta (si el texto ocupa más líneas de las previstas) o deja
   un blanco enorme. Acá el widget avisa cuánto mide con `postMessage` y este
   script ajusta el iframe. Es lo mismo que hacen Flourish y Datawrapper.

   Sin JavaScript en la página que embebe, el iframe igual se dibuja con una
   altura de arranque que cubre el caso peor: se ve todo, sólo que con blanco
   abajo en pantallas anchas.
   =========================================================================== */

(function () {
  "use strict";

  // Alto de arranque, antes del primer aviso del widget. Cubre el caso peor
  // MEDIDO —la versión completa en una columna de celular—, no el de
  // escritorio: de más sólo sobra blanco un instante; de menos, recorta.
  var ALTO_INICIAL = 1860;

  // De dónde salió este script, para resolver rutas relativas del data-src.
  var actual = document.currentScript;
  var base = actual ? actual.src.replace(/embed\.js(\?.*)?$/, "") : "";

  var pendientes = {};
  var contador = 0;

  function montar(div) {
    if (div.dataset.iveMontado === "1") return;
    div.dataset.iveMontado = "1";

    var src = div.getAttribute("data-src") || base;
    var id = "ive-" + (++contador);

    var iframe = document.createElement("iframe");
    iframe.src = src;
    // El `name` viaja al widget y vuelve en el mensaje: con dos widgets en la
    // misma nota, es lo que dice cuál avisó.
    iframe.name = id;
    iframe.title = div.getAttribute("data-title") ||
      "¿Cuál es tu probabilidad de apoyar el derecho a la interrupción voluntaria del embarazo? — El Observador";
    iframe.width = "100%";
    iframe.height = String(ALTO_INICIAL);
    iframe.setAttribute("frameborder", "0");
    iframe.setAttribute("scrolling", "no");
    iframe.setAttribute("loading", "lazy");
    iframe.style.cssText = "display:block;border:none;width:100%;";

    div.appendChild(iframe);
    pendientes[id] = iframe;
  }

  function montarTodos() {
    var divs = document.querySelectorAll(".ive-embed, [data-ive-embed]");
    for (var i = 0; i < divs.length; i++) montar(divs[i]);
  }

  window.addEventListener("message", function (e) {
    var d = e.data;
    if (!d || d.tipo !== "ive-widget:alto") return;
    // El alto viene de un widget que NOSOTROS insertamos; se ignora cualquier
    // otro mensaje, y el valor se acota para que un mensaje raro no estire la
    // página a lo loco.
    var alto = parseInt(d.alto, 10);
    if (!(alto > 0) || alto > 20000) return;

    var iframe = d.id && pendientes[d.id];
    if (!iframe) {
      // Sin id —o con uno desconocido— se busca por la ventana que envió.
      for (var k in pendientes) {
        if (pendientes[k].contentWindow === e.source) { iframe = pendientes[k]; break; }
      }
    }
    if (iframe) iframe.height = String(alto);
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", montarTodos);
  } else {
    montarTodos();
  }
})();
