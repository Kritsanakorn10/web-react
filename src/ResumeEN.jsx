export default function ResumeEN() {
  const contacts = [
    "Nakhon Phanom, Thailand",
    "10kritsanakorn@gmail.com",
    "098-183-8570",
  ];

  const skills = {
    "Data & Analysis": [
      "MATLAB (Deep Learning & Machine Learning Toolbox)",
      "Python (Pandas, NumPy, Scikit-learn)",
      "YOLO, LSTM, SVR",
    ],
    Frontend: ["Angular", "Tailwind CSS", "HTML", "CSS", "TypeScript"],
    Tools: ["VS Code", "Excel", "Jupyter Notebook", "GitHub"],
  };

  const softSkills = [
    "Communication",
    "Teamwork",
    "Problem Solving",
    "Adaptability",
    "Easy to get along with everyone",
    "Continuous Learning",
  ];
  const floaters = [
    { size: "110px", durX: "10s", durY: "13s", delay: "-1s" },
    { size: "95px", durX: "12s", durY: "9s", delay: "-3s" },
    { size: "120px", durX: "11s", durY: "14s", delay: "-5s" },
    { size: "88px", durX: "9s", durY: "12s", delay: "-2s" },
    { size: "105px", durX: "13s", durY: "10s", delay: "-4s" },
    { size: "100px", durX: "12s", durY: "11s", delay: "-6s" },
    { size: "80px", durX: "9s", durY: "8s", delay: "-7s" },
    { size: "130px", durX: "14s", durY: "12s", delay: "-8s" },
  ];

  return (
    <main
      id="top"
      className="relative min-h-screen bg-[#0a0f1f] text-slate-100 font-[Space_Grotesk] overflow-hidden"
    >
      <style>{`
        @keyframes float {
          0% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
          100% { transform: translateY(0px); }
        }
        @keyframes shimmer {
          0% { background-position: 0% 50%; }
          100% { background-position: 200% 50%; }
        }
        @keyframes glowPulse {
          0%, 100% { box-shadow: 0 0 25px rgba(34, 211, 238, 0.25); }
          50% { box-shadow: 0 0 50px rgba(34, 211, 238, 0.5); }
        }
        @keyframes revealUp {
          0% { opacity: 0; transform: translateY(14px); }
          100% { opacity: 1; transform: translateY(0px); }
        }
        @keyframes aboutZoomIn {
          0% { transform: scale(0.94); opacity: 0.7; }
          100% { transform: scale(1); opacity: 1; }
        }
        .card-float { animation: float 6s ease-in-out infinite; }
        .card-float-delay { animation: float 7s ease-in-out infinite 1s; }
        .glow-pulse { animation: glowPulse 4.5s ease-in-out infinite; }
        .shimmer { background-size: 200% 200%; animation: shimmer 8s linear infinite; }
        .reveal-up { animation: revealUp 0.9s ease-out both; }
        .about-zoom {
          transition: transform 0.45s cubic-bezier(0.2, 0.9, 0.3, 1), box-shadow 0.45s ease, border-color 0.45s ease;
          transform-origin: center center;
        }
        .about-zoom-active {
          animation: aboutZoomIn 0.45s cubic-bezier(0.2, 0.9, 0.3, 1);
          transform: scale(1.02);
          border-color: rgba(103, 232, 249, 0.85);
          box-shadow: 0 0 0 1px rgba(103, 232, 249, 0.35), 0 24px 54px rgba(34, 211, 238, 0.3);
        }
        @keyframes dvdX {
          0% { transform: translate3d(0, 0, 0); }
          100% { transform: translate3d(calc(100vw - var(--size)), 0, 0); }
        }
        @keyframes dvdY {
          0% { transform: translate3d(0, 0, 0); }
          100% { transform: translate3d(0, calc(100vh - var(--size)), 0); }
        }
        .dvd-wrap {
          position: absolute;
          inset: 0;
          animation: dvdX var(--durx) linear infinite alternate;
          animation-delay: var(--delay);
        }
        .dvd-float {
          width: var(--size);
          height: var(--size);
          object-fit: contain;
          border-radius: 999px;
          opacity: 0.55;
          filter: drop-shadow(0 0 26px rgba(34, 211, 238, 0.35));
          animation: dvdY var(--dury) linear infinite alternate;
          animation-delay: var(--delay);
        }
      `}</style>
      <div className="pointer-events-none fixed inset-0">
        {floaters.map((item, index) => (
          <div
            key={`${item.size}-${index}`}
            className="dvd-wrap"
            style={{
              "--size": item.size,
              "--durx": item.durX,
              "--dury": item.durY,
              "--delay": item.delay,
            }}
          >
            <img src="/775.png" alt="" className="dvd-float" />
          </div>
        ))}
        <div className="absolute -top-24 left-1/2 h-72 w-72 -translate-x-1/2 rounded-full bg-cyan-400/25 blur-[120px] motion-safe:animate-pulse" />
        <div className="absolute bottom-0 right-0 h-80 w-80 rounded-full bg-lime-300/20 blur-[140px]" />
        <div className="absolute top-1/3 -left-10 h-64 w-64 rounded-full bg-sky-500/20 blur-[110px]" />
      </div>

      <div className="relative max-w-5xl mx-auto px-6 pb-12 pt-24 sm:px-10">
        {/* Header */}
        <header className="relative mb-10 rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur-xl shadow-[0_0_60px_rgba(34,211,238,0.12)] glow-pulse reveal-up">
          <div className="absolute -right-8 -top-8 h-28 w-28 rounded-full border border-white/10 bg-white/5 blur-sm" />
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/80">
                Profile Picture
              </p>
              <h1 className="mt-3 text-4xl font-semibold tracking-tight text-white sm:text-5xl">
                KRITSANAKORN TAPPITAK
              </h1>
              <p className="mt-2 text-sm text-slate-300">Computer Engineering</p>
              <div className="mt-6 h-px w-full bg-gradient-to-r from-cyan-400/0 via-cyan-300/60 to-lime-200/0 shimmer" />
            </div>
            <div className="flex flex-wrap gap-2 text-xs text-slate-200">
              {contacts.map((item) => (
                <span
                  key={item}
                  className="rounded-full border border-white/10 bg-white/10 px-3 py-1 backdrop-blur transition hover:-translate-y-1 hover:border-cyan-300/50 hover:text-cyan-100"
                >
                  {item}
                </span>
              ))}
            </div>
          </div>
        </header>

        <section
          id="about"
          className="mb-8 grid gap-6 lg:grid-cols-[1.2fr_0.8fr] scroll-mt-28"
        >
          {/* Summary */}
          <div
            id="about-card"
            className="rounded-3xl border border-white/10 bg-white/5 p-7 backdrop-blur-xl card-float reveal-up"
          >
            <h2 className="text-sm font-semibold uppercase tracking-[0.3em] text-cyan-300/80">
              About Me
            </h2>
            <p className="mt-4 text-sm leading-relaxed text-slate-200">
              Hello, my name is Kritsanakorn Tappitak, my nickname is "Gram". I
              graduated with a Bachelor's degree in Computer Engineering from Nakhon
              Phanom University. I am interested in data analysis, model development,
              and machine learning. I enjoy all aspects of computer science. I am eager
              to learn new things and continuously seek further knowledge, as well as
              keep up with new technologies. I hope to collaborate with anyone who is
              interested. Thank you.
            </p>
          </div>

          {/* Language */}
          <div className="rounded-3xl border border-white/10 bg-gradient-to-br from-white/10 to-white/5 p-7 backdrop-blur-xl card-float-delay reveal-up">
            <h2 className="text-sm font-semibold uppercase tracking-[0.3em] text-lime-200/90">
              Language
            </h2>
            <div className="mt-4 space-y-3 text-sm text-slate-200">
              <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/10 px-4 py-3">
                <span>Thai (Native)</span>
                <span className="text-xs text-lime-200/80">Native</span>
              </div>
              <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/10 px-4 py-3">
                <span>English (Intermediate)</span>
                <span className="text-xs text-cyan-200/80">Intermediate</span>
              </div>
            </div>
          </div>
        </section>

        {/* Skills */}
        <section
          id="skills"
          className="mb-8 rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur-xl reveal-up scroll-mt-28"
        >
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold uppercase tracking-[0.3em] text-cyan-300/80">
              Technical Skills
            </h2>
            <span className="text-xs text-slate-400">Stack map</span>
          </div>
          <div className="mt-6 grid gap-6 md:grid-cols-3">
            {Object.entries(skills).map(([group, items]) => (
              <div
                key={group}
                className="rounded-2xl border border-white/10 bg-white/5 p-5 transition hover:-translate-y-1 hover:border-cyan-300/50 hover:bg-white/10"
              >
                <p className="text-xs font-semibold uppercase tracking-[0.25em] text-slate-300">
                  {group}
                </p>
                <ul className="mt-4 space-y-2 text-sm text-slate-200">
                  {items.map((item) => (
                    <li key={item} className="flex items-start gap-2">
                      <span className="mt-1 h-2 w-2 rounded-full bg-cyan-300" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        <section className="mb-8 grid gap-6 lg:grid-cols-[1fr_1fr] scroll-mt-28">
          {/* Soft Skills */}
          <div className="rounded-3xl border border-white/10 bg-white/5 p-7 backdrop-blur-xl reveal-up">
            <h2 className="text-sm font-semibold uppercase tracking-[0.3em] text-cyan-300/80">
              Soft Skills
            </h2>
            <div className="mt-4 flex flex-wrap gap-2 text-sm text-slate-200">
              {softSkills.map((item) => (
                <span
                  key={item}
                  className="rounded-full border border-white/10 bg-white/10 px-3 py-1 text-xs transition hover:-translate-y-1 hover:border-lime-200/60 hover:text-lime-100"
                >
                  {item}
                </span>
              ))}
            </div>
          </div>

          {/* Education */}
          <div
            id="education"
            className="rounded-3xl border border-white/10 bg-gradient-to-br from-white/10 to-white/5 p-7 backdrop-blur-xl reveal-up scroll-mt-28"
          >
            <h2 className="text-sm font-semibold uppercase tracking-[0.3em] text-lime-200/90">
              Education
            </h2>
            <div className="mt-4 space-y-3 text-sm text-slate-200">
              <p className="text-base font-medium text-white">
                Bachelor of Computer Engineering
              </p>
              <p>Nakhon Phanom University</p>
              <p>2021 - 2025</p>
              <p className="text-xs text-slate-300">GPA: 2.74</p>
            </div>
          </div>
        </section>

        {/* Experience */}
        <section
          id="experience"
          className="mb-8 rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur-xl reveal-up scroll-mt-28"
        >
          <h2 className="text-sm font-semibold uppercase tracking-[0.3em] text-cyan-300/80">
            Experience
          </h2>
          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-6 transition hover:-translate-y-1 hover:border-cyan-300/50 hover:bg-white/10">
              <h3 className="text-base font-medium text-white">Internship</h3>
              <p className="mt-2 text-sm text-slate-300">2024</p>
              <p className="mt-2 text-sm text-slate-200">
                NT Nakhon Phanom - National Telecom Public Company Limited (1 month)
              </p>
              <ul className="mt-4 space-y-2 text-sm text-slate-200">
                <li>
                  Install and maintain network equipment (Routers, LAN, LAN
                  switches) for government agencies (schools, hospitals, local
                  administrative organizations).
                </li>
              </ul>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-6 transition hover:-translate-y-1 hover:border-cyan-300/50 hover:bg-white/10">
              <h3 className="text-base font-medium text-white">Projects</h3>
              <p className="mt-2 text-sm text-slate-300">2024 - 2025</p>
              <p className="mt-2 text-sm text-slate-200">
                Final Project: Rainfall Prediction Model Using Machine Learning Techniques
                Case Study: Nakhon Phanom Province
              </p>
              <ul className="mt-4 space-y-2 text-sm text-slate-200">
                <li>
                  Graduation Project: Developed rainfall forecasting models (LSTM, SVR)
                  in MATLAB; achieved 76.28% accuracy with LSTM using Thai Meteorological
                  Department evaluation criteria.
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Footer */}
      </div>
    </main>
  );
}

