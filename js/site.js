/* William Tanna — shared site behaviors (Step 1) */

(function () {
  "use strict";

  function qs(sel, root) {
    return (root || document).querySelector(sel);
  }

  function qsa(sel, root) {
    return Array.from((root || document).querySelectorAll(sel));
  }

  /* —— Page transitions —— */
  window.navigateWithAnimation = function (event, url) {
    if (event) event.preventDefault();
    const overlay = qs("#pageTransition");
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (overlay && !reduced) {
      overlay.classList.add("active");
      overlay.setAttribute("aria-hidden", "false");
      setTimeout(function () {
        window.location.href = url;
      }, 580);
    } else if (overlay && reduced) {
      window.location.href = url;
    } else {
      window.location.href = url;
    }
  };

  /* —— Nav —— */
  function initNav() {
    const toggle = qs("#siteNavToggle");
    const links = qs("#siteNavLinks");
    const backdrop = qs("#siteNavBackdrop");
    const nav = qs(".site-nav");

    function closeMenu() {
      if (!links) return;
      links.classList.remove("is-open");
      if (toggle) toggle.setAttribute("aria-expanded", "false");
      if (backdrop) backdrop.classList.remove("is-open");
    }

    function openMenu() {
      if (!links) return;
      links.classList.add("is-open");
      if (toggle) toggle.setAttribute("aria-expanded", "true");
      if (backdrop) backdrop.classList.add("is-open");
    }

    window.toggleMenu = function () {
      if (!links) return;
      if (links.classList.contains("is-open")) closeMenu();
      else openMenu();
    };

    if (toggle) {
      toggle.addEventListener("click", function (e) {
        e.stopPropagation();
        window.toggleMenu();
      });
    }

    if (backdrop) backdrop.addEventListener("click", closeMenu);

    document.addEventListener("click", function (e) {
      if (!nav || !links) return;
      if (!nav.contains(e.target) && !links.contains(e.target)) closeMenu();
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") closeMenu();
    });

    if (nav) {
      const onScroll = function () {
        nav.classList.toggle("is-scrolled", window.scrollY > 12);
      };
      window.addEventListener("scroll", onScroll, { passive: true });
      onScroll();
    }
  }

  /* —— Scroll reveals —— */
  function initScrollReveals() {
    const els = qsa(".animate-on-scroll");
    if (!els.length) return;

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      els.forEach(function (el) {
        el.classList.add("visible");
      });
      return;
    }

    const observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
    );

    els.forEach(function (el) {
      observer.observe(el);
    });
  }

  /* —— Floating chat panel shell (wired in Step 2) —— */
  window.initChatShell = function (opts) {
    opts = opts || {};
    const fab = qs(opts.fab || "#chatFab");
    const panel = qs(opts.panel || "#chatPanel");
    const closeBtn = qs(opts.close || "#chatPanelClose");
    if (!fab || !panel) return;

    function open() {
      panel.classList.add("is-open");
      fab.setAttribute("aria-expanded", "true");
    }

    function close() {
      panel.classList.remove("is-open");
      fab.setAttribute("aria-expanded", "false");
    }

    fab.addEventListener("click", function () {
      if (panel.classList.contains("is-open")) close();
      else open();
    });

    if (closeBtn) closeBtn.addEventListener("click", close);
  };

  document.addEventListener("DOMContentLoaded", function () {
    initNav();
    initScrollReveals();
  });
})();
