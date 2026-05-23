/**

 * Veritas AI — Frontend interactions

 * UI-only: does not affect Flask prediction logic

 */



(function () {

  'use strict';



  /** Mark UI ready — content visible even if animations fail */

  function markUiReady() {

    document.documentElement.classList.add('ui-ready');

  }



  /* ---- Particle background (non-blocking, optional) ---- */

  function initParticles() {

    const canvas = document.getElementById('particles');

    if (!canvas) return;



    try {

      const ctx = canvas.getContext('2d');

      if (!ctx) return;



      let particles = [];

      let animId = null;



      function resize() {

        const w = window.innerWidth || 800;

        const h = window.innerHeight || 600;

        canvas.width = w;

        canvas.height = h;

      }



      function createParticles(count) {

        particles = [];

        const n = Math.min(60, Math.max(20, count));

        for (let i = 0; i < n; i++) {

          particles.push({

            x: Math.random() * canvas.width,

            y: Math.random() * canvas.height,

            r: Math.random() * 1.5 + 0.5,

            dx: (Math.random() - 0.5) * 0.3,

            dy: (Math.random() - 0.5) * 0.3,

            opacity: Math.random() * 0.35 + 0.08

          });

        }

      }



      function draw() {

        if (!canvas.isConnected) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);



        particles.forEach((p, i) => {

          ctx.beginPath();

          ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);

          ctx.fillStyle = 'rgba(34, 211, 238, ' + p.opacity + ')';

          ctx.fill();



          p.x += p.dx;

          p.y += p.dy;

          if (p.x < 0 || p.x > canvas.width) p.dx *= -1;

          if (p.y < 0 || p.y > canvas.height) p.dy *= -1;



          for (let j = i + 1; j < particles.length; j++) {

            const p2 = particles[j];

            const dist = Math.hypot(p.x - p2.x, p.y - p2.y);

            if (dist < 100) {

              ctx.beginPath();

              ctx.moveTo(p.x, p.y);

              ctx.lineTo(p2.x, p2.y);

              ctx.strokeStyle = 'rgba(59, 130, 246, ' + (0.06 * (1 - dist / 100)) + ')';

              ctx.stroke();

            }

          }

        });



        animId = requestAnimationFrame(draw);

      }



      resize();

      createParticles(Math.floor(window.innerWidth / 18));

      draw();



      window.addEventListener('resize', function () {

        resize();

        createParticles(Math.floor(window.innerWidth / 18));

      });

    } catch (err) {

      console.warn('[Veritas] Particle background disabled:', err);

      canvas.style.display = 'none';

    }

  }



  function initSidebar() {

    const sidebar = document.getElementById('sidebar');

    const toggle = document.getElementById('sidebarToggle');

    const navLinks = document.querySelectorAll('.sidebar-nav .nav-link');



    if (toggle && sidebar) {

      toggle.addEventListener('click', function () {

        sidebar.classList.toggle('open');

      });

    }



    navLinks.forEach(function (link) {

      link.addEventListener('click', function () {

        navLinks.forEach(function (l) { l.classList.remove('active'); });

        link.classList.add('active');

        if (sidebar) sidebar.classList.remove('open');

      });

    });



    if (typeof IntersectionObserver === 'undefined') return;



    const sections = document.querySelectorAll('section[id], header[id]');

    const observer = new IntersectionObserver(

      function (entries) {

        entries.forEach(function (entry) {

          if (entry.isIntersecting) {

            const id = entry.target.id;

            navLinks.forEach(function (link) {

              link.classList.toggle('active', link.getAttribute('href') === '#' + id);

            });

          }

        });

      },

      { rootMargin: '-20% 0px -55% 0px', threshold: 0 }

    );



    sections.forEach(function (s) { observer.observe(s); });

  }



  function initCharCount() {

    const textarea = document.getElementById('news');

    const counter = document.getElementById('charCount');

    if (!textarea || !counter) return;



    function update() {

      const len = textarea.value.length;

      counter.textContent = len + ' character' + (len !== 1 ? 's' : '');

    }



    textarea.addEventListener('input', update);

    update();

  }



  function initFormLoading() {

    const form = document.getElementById('predictForm');

    const btn = document.getElementById('predictBtn');

    if (!form || !btn) return;



    form.addEventListener('submit', function () {

      btn.classList.add('loading');

    });



    /* Reset loading state when page reloads with results */

    btn.classList.remove('loading');

  }



  function initConfidenceMeter() {

    const fill = document.querySelector('.confidence-fill');

    if (!fill) return;



    const target = parseInt(fill.getAttribute('data-target'), 10) || 0;

    requestAnimationFrame(function () {

      setTimeout(function () {

        fill.style.width = target + '%';

      }, 120);

    });

  }



  function initCharts() {

    if (typeof Chart === 'undefined') return;



    try {

      Chart.defaults.color = '#8b9dc9';

      Chart.defaults.borderColor = 'rgba(99, 179, 237, 0.12)';

      Chart.defaults.font.family = "'Inter', sans-serif";



      const gridColor = 'rgba(99, 179, 237, 0.08)';

      const confusionEl = document.getElementById('confusionChart');

      const comparisonEl = document.getElementById('comparisonChart');



      if (confusionEl) {

        new Chart(confusionEl, {

          type: 'bar',

          data: {

            labels: ['Pred Fake', 'Pred Real'],

            datasets: [

              {

                label: 'Actual Fake',

                data: [420, 38],

                backgroundColor: 'rgba(239, 68, 68, 0.7)',

                borderRadius: 6

              },

              {

                label: 'Actual Real',

                data: [45, 497],

                backgroundColor: 'rgba(16, 185, 129, 0.7)',

                borderRadius: 6

              }

            ]

          },

          options: {

            responsive: true,

            maintainAspectRatio: false,

            scales: {

              x: { grid: { color: gridColor } },

              y: { grid: { color: gridColor }, beginAtZero: true }

            }

          }

        });

      }



      if (comparisonEl) {

        new Chart(comparisonEl, {

          type: 'radar',

          data: {

            labels: ['Accuracy', 'Precision', 'Recall', 'F1', 'Speed'],

            datasets: [

              {

                label: 'Current Model',

                data: [92, 91, 90, 90, 95],

                borderColor: '#22d3ee',

                backgroundColor: 'rgba(34, 211, 238, 0.15)',

                pointBackgroundColor: '#22d3ee'

              },

              {

                label: 'Baseline',

                data: [78, 76, 74, 75, 88],

                borderColor: '#8b5cf6',

                backgroundColor: 'rgba(139, 92, 246, 0.1)',

                pointBackgroundColor: '#8b5cf6'

              }

            ]

          },

          options: {

            responsive: true,

            maintainAspectRatio: false,

            scales: {

              r: {

                angleLines: { color: gridColor },

                grid: { color: gridColor },

                suggestedMin: 0,

                suggestedMax: 100

              }

            }

          }

        });

      }

    } catch (err) {

      console.warn('[Veritas] Chart init skipped:', err);

    }

  }



  function boot() {

    markUiReady();



    try { initSidebar(); } catch (e) { console.warn(e); }

    try { initCharCount(); } catch (e) { console.warn(e); }

    try { initFormLoading(); } catch (e) { console.warn(e); }

    try { initConfidenceMeter(); } catch (e) { console.warn(e); }

    try { initCharts(); } catch (e) { console.warn(e); }



    /* Defer particles so main UI paints first */

    if (window.requestIdleCallback) {

      requestIdleCallback(function () { initParticles(); }, { timeout: 500 });

    } else {

      setTimeout(initParticles, 50);

    }

  }



  if (document.readyState === 'loading') {

    document.addEventListener('DOMContentLoaded', boot);

  } else {

    boot();

  }

})();


