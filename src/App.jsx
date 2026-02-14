import { useEffect, useRef, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import ResumeEN from "./ResumeEN";
import ResumeTH from "./ResumeTH";
import Projects from "./Projects";
// import MLToolkit from "./MLToolkit";
import Hbdweb from "./Hbdweb";
import VideoDownloader from "./VideoDownloader";
import TextArt from "./TextArt";

function SocialRail({ isHbdweb }) {
  const socials = [
    {
      id: "github",
      href: "https://github.com/Kritsanakorn10/MyProjectKritsanakorn",
      label: "GitHub",
      icon: "M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.34-3.369-1.34-.454-1.152-1.11-1.458-1.11-1.458-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12c0-5.523-4.477-10-10-10z",
      tones: isHbdweb
        ? "bg-slate-200/95 text-slate-900 border-slate-400/60"
        : "bg-slate-900/70 text-white border-cyan-300/30",
    },
    {
      id: "facebook",
      href: "https://www.facebook.com/kritsanakorn.tappitak.gram",
      label: "Facebook",
      icon: "M22 12c0-5.523-4.477-10-10-10S2 6.477 2 12c0 4.991 3.657 9.128 8.438 9.878v-6.987h-2.54V12h2.54V9.797c0-2.506 1.492-3.89 3.777-3.89 1.094 0 2.238.195 2.238.195v2.46h-1.26c-1.243 0-1.63.771-1.63 1.562V12h2.773l-.443 2.89h-2.33v6.988C18.343 21.128 22 16.991 22 12z",
      tones: isHbdweb
        ? "bg-blue-100/95 text-blue-700 border-blue-300/70"
        : "bg-slate-900/70 text-blue-100 border-cyan-300/30",
    },
    {
      id: "instagram",
      href: "https://www.instagram.com/gramksnk/",
      label: "Instagram",
      icon: "M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z",
      tones: isHbdweb
        ? "bg-pink-50/95 text-pink-700 border-pink-300/70"
        : "bg-slate-900/70 text-pink-100 border-cyan-300/30",
    },
  ];

  return (
    <div className="fixed bottom-6 left-6 z-[130] flex flex-col gap-2 sm:bottom-8 sm:left-8 lg:left-10">
      {socials.map((social) => (
        <a
          key={social.id}
          href={social.href}
          target="_blank"
          rel="noreferrer"
          className={`group flex h-12 w-12 items-center justify-start overflow-hidden rounded-full border px-3 shadow-xl backdrop-blur-md transition-all duration-300 hover:w-40 ${social.tones}`}
          aria-label={social.label}
        >
          <svg viewBox="0 0 24 24" className="h-6 w-6 shrink-0" fill="currentColor">
            <path d={social.icon} />
          </svg>
          <span className="ml-3 max-w-0 translate-x-2 overflow-hidden whitespace-nowrap text-xs font-semibold tracking-wide opacity-0 transition-all duration-300 group-hover:max-w-24 group-hover:translate-x-0 group-hover:opacity-100">
            {social.label}
          </span>
        </a>
      ))}
    </div>
  );
}



function Navbar({ lang, setLang, onToggleNav, onOpenSection, isHbdweb }) {
  const navItems = [
    { id: "about", label: "About Me" },
    { id: "skills", label: "Skills" },
    { id: "experience", label: "Experience" },
    { id: "education", label: "Education" },
  ];

  const shellClass = isHbdweb
    ? "w-[186px] flex-col gap-2 rounded-[22px] border border-sky-200/90 bg-white/90 p-3 text-slate-700 shadow-[0_12px_32px_rgba(14,116,144,0.18)]"
    : "w-full max-w-5xl flex-col gap-3 rounded-[28px] border border-white/10 bg-slate-950/55 p-4 text-slate-100 sm:flex-row sm:items-center sm:justify-between";

  const chipClass = isHbdweb
    ? "border-sky-200 bg-sky-50/95 text-slate-700 hover:border-sky-400 hover:bg-sky-100"
    : "border-white/10 bg-white/5 text-slate-200 hover:border-cyan-300/70 hover:text-cyan-100";

  return (
    <div className={`flex ${isHbdweb ? "h-full items-center justify-start pl-4" : "justify-center px-4 pt-5 sm:px-8"}`}>
      <div className={`nav-shell nav-sheen flex shadow-2xl backdrop-blur-xl ${shellClass}`}>
        <Link
          to="/"
          onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
          className={`nav-chip inline-flex items-center justify-center rounded-full border font-semibold uppercase ${
            isHbdweb ? "h-9 gap-1.5 px-3 text-[10px] tracking-[0.14em]" : "h-11 gap-2 px-4 text-[11px] tracking-[0.2em]"
          } ${chipClass}`}
        >
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 10.5l9-7 9 7M5 9.5V20h14V9.5" />
          </svg>
          Home
        </Link>

        <div className={`flex ${isHbdweb ? "flex-col" : "flex-wrap items-center justify-center"} gap-2`}>
          {!isHbdweb &&
            navItems.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => onOpenSection(item.id)}
                className={`nav-chip h-11 rounded-full border px-4 text-[11px] font-semibold uppercase tracking-[0.16em] ${chipClass}`}
              >
                {item.label}
              </button>
            ))}
          <NavLink
            to="/projects"
            className={({ isActive }) =>
              `nav-chip inline-flex items-center justify-center rounded-full border font-semibold uppercase ${
                isHbdweb ? "h-9 px-3 text-[10px] tracking-[0.14em]" : "h-11 px-4 text-[11px] tracking-[0.16em]"
              } ${
                isActive
                  ? isHbdweb
                    ? "border-sky-500 bg-sky-500/20 text-sky-800"
                    : "border-cyan-300 bg-cyan-400/20 text-cyan-100"
                  : chipClass
              }`
            }
          >
            Project
          </NavLink>
        </div>

        <div className={`flex items-center ${isHbdweb ? "justify-between gap-1.5" : "gap-2"}`}>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setLang("EN")}
              className={`nav-chip rounded-full border font-bold ${
                isHbdweb ? "h-8 px-2.5 text-[9px] tracking-[0.1em]" : "h-9 px-3 text-[10px] tracking-[0.12em]"
              } ${
                lang === "EN"
                  ? isHbdweb
                    ? "border-sky-500 bg-sky-500 text-white"
                    : "border-cyan-200 bg-cyan-500 text-white"
                  : chipClass
              }`}
            >
              EN
            </button>
            <button
              type="button"
              onClick={() => setLang("TH")}
              className={`nav-chip rounded-full border font-bold ${
                isHbdweb ? "h-8 px-2.5 text-[9px] tracking-[0.1em]" : "h-9 px-3 text-[10px] tracking-[0.12em]"
              } ${
                lang === "TH"
                  ? isHbdweb
                    ? "border-sky-500 bg-sky-500 text-white"
                    : "border-cyan-200 bg-cyan-500 text-white"
                  : chipClass
              }`}
            >
              TH
            </button>
          </div>
          <button
            type="button"
            onClick={onToggleNav}
            className={`nav-chip inline-flex items-center justify-center rounded-full border ${
              isHbdweb ? "h-8 w-8" : "ml-2 h-9 w-9"
            } ${
              isHbdweb
                ? "border-rose-300 bg-rose-50 text-rose-500 hover:bg-rose-100"
                : "border-rose-300/40 bg-rose-500/20 text-rose-100"
            }`}
            aria-label="Hide navbar"
            title="Hide navbar"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2.4">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

const SECTION_MODAL_COPY = {
  EN: {
    about: {
      eyebrow: "About Me",
      title: "Kritsanakorn Tappitak",
      body:
        "Hello, my name is Kritsanakorn Tappitak, my nickname is \"Gram\". I graduated with a Bachelor's degree in Computer Engineering from Nakhon Phanom University. I am interested in data analysis, model development, and machine learning. I enjoy all aspects of computer science. I am eager to learn new things and continuously seek further knowledge, as well as keep up with new technologies. I hope to collaborate with anyone who is interested. Thank you.",
    },
    skills: {
      eyebrow: "Skills",
      title: "Technical Skills",
      body:
        "MATLAB, Python (Pandas, NumPy, Scikit-learn), YOLO, LSTM, SVR, Angular, Tailwind CSS, TypeScript, VS Code, GitHub, and Jupyter Notebook.",
    },
    experience: {
      eyebrow: "Experience",
      title: "Experience",
      body:
        "Internship at NT Nakhon Phanom (2024): installed and maintained network equipment for government agencies. Final-year project (2024-2025): rainfall prediction using LSTM and SVR in MATLAB.",
    },
    education: {
      eyebrow: "Education",
      title: "Bachelor of Computer Engineering",
      body: "Nakhon Phanom University, 2021-2025, GPA 2.74.",
    },
  },
  TH: {
    about: {
      eyebrow: "เกี่ยวกับฉัน",
      title: "กฤษณกร เทพพิทักษ์",
      body:
        "สวัสดีครับ ผมชื่อ กฤษณกร เทพพิทักษ์ ชื่อเล่น \"แกรม\" ผมจบการศึกษาปริญญาตรีคณะ วิศวกรรมคอมพิวเตอร์จากมหาวิทยาลัยนครพนม ผมสนใจด้านการวิเคราะห์ข้อมูล การพัฒนาโมเดล และแมชชีนเลิร์นนิง ผมชอบทุกอย่างที่เกี่ยวกับวิทยาการคอมพิวเตอร์ ผมชอบที่จะเรียนรู้ สิ่งใหม่ๆ และหาความรู้เพิ่มเติมอย่างต่อเนื่อง และอัพเดตเทคโนโลยีใหม่ๆอยู่เสมอ และผมหวัง ที่จะร่วมงานกับทุกท่านที่สนใจในตัวผม ขอบคุณครับ",
    },
    skills: {
      eyebrow: "ทักษะ",
      title: "ทักษะเชิงเทคนิค",
      body:
        "MATLAB, Python (Pandas, NumPy, Scikit-learn), YOLO, LSTM, SVR, Angular, Tailwind CSS, TypeScript, VS Code, GitHub และ Jupyter Notebook",
    },
    experience: {
      eyebrow: "ประสบการณ์",
      title: "ฝึกงานและโปรเจกต์",
      body:
        "ฝึกงานที่ NT นครพนม ในปี 2024 และทำโครงงานจบช่วงปี 2024-2025 เกี่ยวกับการพยากรณ์ปริมาณน้ำฝนด้วย LSTM และ SVR บน MATLAB",
    },
    education: {
      eyebrow: "การศึกษา",
      title: "ปริญญาตรี วิศวกรรมคอมพิวเตอร์",
      body: "มหาวิทยาลัยนครพนม, 2021-2025, GPA 2.74",
    },
  },
};

const SKILL_MODAL_GROUPS = {
  EN: [
    {
      title: "Data & Analysis",
      items: [
        "MATLAB (Deep Learning & Machine Learning Toolbox)",
        "Python (Pandas, NumPy, Scikit-learn)",
        "YOLO, LSTM, SVR",
      ],
    },
    {
      title: "Frontend",
      items: ["Angular", "Tailwind CSS", "HTML", "CSS", "TypeScript"],
    },
    {
      title: "Tools",
      items: ["VS Code", "Excel", "Jupyter Notebook", "GitHub"],
    },
  ],
  TH: [
    {
      title: "ข้อมูลและการวิเคราะห์",
      items: [
        "MATLAB",
        "Python (Pandas, NumPy, Scikit-learn)",
        "YOLO, LSTM, SVR",
      ],
    },
    {
      title: "Frontend",
      items: ["Angular", "Tailwind CSS", "HTML", "CSS", "TypeScript"],
    },
    {
      title: "Tools",
      items: ["VS Code", "Git, GitHub", "Jupyter Notebook"],
    },
  ],
};

const SOFT_SKILLS = {
  EN: [
    "Communication",
    "Teamwork",
    "Problem Solving",
    "Adaptability",
    "Easy to get along with everyone",
    "Continuous Learning",
  ],
  TH: [
    "ทักษะการสื่อสาร",
    "การทำงานเป็นทีม",
    "การแก้ปัญหา",
    "ความสามารถในการปรับตัว",
    "เข้ากับทุกคนได้ง่าย",
    "การเรียนรู้อย่างต่อเนื่อง",
  ],
};

const EXPERIENCE_MODAL_ITEMS = {
  EN: [
    {
      title: "Internship",
      period: "2024",
      summary: "NT Nakhon Phanom - National Telecom Public Company Limited (1 month)",
      details:
        "Installed and maintained network equipment (routers, LAN, switches) for government agencies such as schools, hospitals, and local administrative organizations.",
    },
    {
      title: "Projects",
      period: "2024 - 2025",
      summary:
        "Final Project: Rainfall Prediction Model Using Machine Learning Techniques (Case Study: Nakhon Phanom Province)",
      details:
        "Developed rainfall forecasting models (LSTM, SVR) in MATLAB; achieved 76.28% accuracy with LSTM under Thai Meteorological Department evaluation criteria.",
    },
  ],
  TH: [
    {
      title: "ฝึกงาน",
      period: "2024",
      summary: "NT นครพนม - บริษัท โทรคมนาคมแห่งชาติ จำกัด (มหาชน) (1 เดือน)",
      details:
        "ติดตั้งและบำรุงรักษาอุปกรณ์เครือข่าย (Routers, LAN, LAN switches) สำหรับหน่วยงานภาครัฐ เช่น โรงเรียน โรงพยาบาล และองค์กรปกครองส่วนท้องถิ่น",
    },
    {
      title: "โปรเจกต์",
      period: "2024 - 2025",
      summary:
        "โครงงานจบการศึกษา: แบบจำลองการพยากรณ์ปริมาณน้ำฝนด้วยเทคนิคการเรียนรู้แบบเครื่อง (กรณีศึกษา: จังหวัดนครพนม)",
      details:
        "พัฒนาแบบจำลองพยากรณ์ฝน (LSTM, SVR) ใน MATLAB และได้ความแม่นยำ 76.28% ด้วย LSTM ตามเกณฑ์การประเมินของกรมอุตุนิยมวิทยา",
    },
  ],
};

export default function App() {
  const [lang, setLang] = useState("EN");
  const [navHidden, setNavHidden] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [activeModalSection, setActiveModalSection] = useState(null);
  const [modalVisible, setModalVisible] = useState(false);
  const closeTimerRef = useRef(null);
  const location = useLocation();
  const navigate = useNavigate();
  const isHbdweb = location.pathname === "/projects/happy";

  const showNavbar = !navHidden || isHovered;
  const sectionCopy = activeModalSection ? SECTION_MODAL_COPY[lang][activeModalSection] : null;
  const skillGroups = SKILL_MODAL_GROUPS[lang];
  const softSkills = SOFT_SKILLS[lang];
  const experienceItems = EXPERIENCE_MODAL_ITEMS[lang];

  useEffect(() => {
    return () => {
      if (closeTimerRef.current) {
        clearTimeout(closeTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!activeModalSection) return;
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setModalVisible(false);
        closeTimerRef.current = setTimeout(() => {
          setActiveModalSection(null);
          closeTimerRef.current = null;
        }, 220);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeModalSection]);

  useEffect(() => {
    if (!activeModalSection) return undefined;
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, [activeModalSection]);

  const showSectionModal = (sectionId) => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
    setActiveModalSection(sectionId);
    setModalVisible(false);
    requestAnimationFrame(() => setModalVisible(true));
  };

  const closeSectionModal = () => {
    if (!activeModalSection) return;
    setModalVisible(false);
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
    }
    closeTimerRef.current = setTimeout(() => {
      setActiveModalSection(null);
      closeTimerRef.current = null;
    }, 220);
  };

  const openSectionModal = (sectionId) => {
    if (location.pathname !== "/") {
      navigate("/");
      setTimeout(() => showSectionModal(sectionId), 0);
      return;
    }
    showSectionModal(sectionId);
  };

  return (
    <main className={`min-h-screen transition-colors duration-700 ${isHbdweb ? "bg-slate-100" : "bg-[#020617]"}`}>
      <div
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className={`fixed z-[120] transition-all duration-300 ${
          isHbdweb ? "inset-y-0 left-0 w-56" : "inset-x-0 top-0 h-32"
        }`}
      >
        <div
          className={`absolute inset-0 transition-all duration-500 ease-out ${
            showNavbar
              ? "pointer-events-auto translate-y-0 translate-x-0 opacity-100"
              : `pointer-events-none opacity-0 ${isHbdweb ? "-translate-x-full" : "-translate-y-full"}`
          }`}
        >
          <Navbar
            lang={lang}
            setLang={setLang}
            onOpenSection={openSectionModal}
            isHbdweb={isHbdweb}
            onToggleNav={() => {
              setNavHidden(true);
              setIsHovered(false);
            }}
          />
        </div>

        <div
          className={`fixed z-[110] transition-all duration-500 ${
            isHbdweb ? "left-6 top-1/2 -translate-y-1/2" : "left-1/2 top-4 -translate-x-1/2"
          } ${navHidden && !isHovered ? "scale-100 opacity-100" : "pointer-events-none scale-75 opacity-0"}`}
        >
          <button
            type="button"
            onClick={() => setNavHidden(false)}
            className={`group relative flex h-12 w-12 items-center justify-center rounded-2xl border backdrop-blur-md transition-all hover:scale-105 active:scale-90 ${
              isHbdweb
                ? "border-slate-400/80 bg-white/90 text-slate-800"
                : "border-cyan-300/40 bg-slate-900/70 text-cyan-100"
            }`}
            aria-label="Show navbar"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2.3">
              <path d={isHbdweb ? "M9 6l6 6-6 6" : "M6 9l6 6 6-6"} />
            </svg>
            <span className="pointer-events-none absolute left-full ml-3 rounded-lg bg-slate-900 px-2 py-1 text-[10px] text-white opacity-0 transition group-hover:opacity-100">
              Show menu
            </span>
          </button>
        </div>
      </div>

      <SocialRail isHbdweb={isHbdweb} />

      <div className="relative z-0">
        <Routes>
          <Route path="/" element={lang === "EN" ? <ResumeEN /> : <ResumeTH />} />
          <Route path="/projects" element={<Projects lang={lang} />} />
          {/* MLToolkit route is intentionally disabled for Render deploy (yt-dlp only). */}
          {/* <Route path="/projects/ml-toolkit" element={<MLToolkit />} /> */}
          <Route path="/projects/video-downloader" element={<VideoDownloader />} />
          <Route path="/projects/happy" element={<Hbdweb />} />
          <Route path="/projects/text-art" element={<TextArt lang={lang} />} />
        </Routes>
      </div>

      {activeModalSection && sectionCopy && (
        <div
          className={`fixed inset-0 z-[250] flex items-center justify-center bg-slate-950/75 px-4 backdrop-blur-sm transition-opacity duration-200 ease-out ${
            modalVisible ? "opacity-100" : "opacity-0"
          }`}
          onClick={closeSectionModal}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              closeSectionModal();
            }
          }}
          aria-label="Close section modal"
        >
          <div
            className={`w-full rounded-3xl border border-cyan-300/40 bg-slate-900/95 p-6 text-slate-100 shadow-[0_20px_80px_rgba(34,211,238,0.25)] transition-all duration-200 ease-out sm:p-8 ${
              activeModalSection === "skills"
                ? "max-w-6xl"
                : activeModalSection === "experience"
                  ? "max-w-4xl"
                  : "max-w-2xl"
            } ${modalVisible ? "scale-100 opacity-100" : "scale-95 opacity-0"}`}
            onClick={(event) => event.stopPropagation()}
          >
            <p className="text-xs uppercase tracking-[0.3em] text-cyan-300/80">{sectionCopy.eyebrow}</p>
            <h2 className="mt-3 text-2xl font-semibold text-white sm:text-3xl">{sectionCopy.title}</h2>

            {activeModalSection === "skills" ? (
              <div className="mt-6">
                <div className="mb-4 flex items-center justify-between text-xs text-slate-400">
                  <span className="uppercase tracking-[0.25em] text-cyan-300/85">
                    {lang === "TH" ? "ทักษะเชิงเทคนิค" : "Technical Skills"}
                  </span>
                  <span>{lang === "TH" ? "ภาพรวมทักษะ" : "Stack map"}</span>
                </div>
                <div className="grid gap-4 md:grid-cols-3">
                  {skillGroups.map((group) => (
                    <div key={group.title} className="rounded-2xl border border-white/10 bg-white/[0.04] p-5">
                      <h3 className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-200">
                        {group.title}
                      </h3>
                      <ul className="mt-4 space-y-2.5 text-sm text-slate-100">
                        {group.items.map((item) => (
                          <li key={item} className="flex items-start gap-2">
                            <span className="mt-1.5 h-2.5 w-2.5 rounded-full bg-cyan-300" />
                            <span>{item}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>

                <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.04] p-5">
                  <h3 className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-200">
                    {lang === "TH" ? "ทักษะการทำงาน" : "Soft Skills"}
                  </h3>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {softSkills.map((item) => (
                      <span
                        key={item}
                        className="rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs text-slate-100"
                      >
                        {item}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ) : activeModalSection === "experience" ? (
              <div className="mt-6 grid gap-4 md:grid-cols-2">
                {experienceItems.map((item) => (
                  <div key={item.title} className="rounded-2xl border border-white/10 bg-white/[0.04] p-5">
                    <h3 className="text-base font-semibold text-white">{item.title}</h3>
                    <p className="mt-1 text-xs text-cyan-200/90">{item.period}</p>
                    <p className="mt-3 text-sm text-slate-200">{item.summary}</p>
                    <p className="mt-3 text-sm leading-relaxed text-slate-300">{item.details}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-5 text-sm leading-relaxed text-slate-200 sm:text-base">{sectionCopy.body}</p>
            )}

            <p className="mt-6 text-xs text-slate-300">
              {lang === "TH" ? "กด Esc หรือคลิกพื้นที่ว่างเพื่อปิด" : "Press Esc or click outside to close"}
            </p>
          </div>
        </div>
      )}
    </main>
  );
}
