import { Link } from "react-router-dom";
// eslint-disable-next-line no-unused-vars
import { motion } from "motion/react";

export default function Projects({ lang = "EN" }) {
  const isEN = lang === "EN";

  const cards = [
    // ML Toolkit card is intentionally hidden for Render deploy (yt-dlp only).
    {
      title: isEN ? "HappyBirthday Web" : "เว็บอวยพรวันเกิด",
      desc: isEN
        ? "Create fullscreen birthday scenes with media background and interactive effects."
        : "สร้างหน้าวันเกิดแบบเต็มจอ พร้อมพื้นหลังสื่อและเอฟเฟกต์โต้ตอบ",
      to: "/projects/happy",
      tag: isEN ? "Interactive" : "โต้ตอบ",
    },
    {
      title: isEN ? "Video Downloader" : "ดาวน์โหลดวิดีโอ",
      desc: isEN
        ? "Paste link, auto-check source details, and download in your preferred format."
        : "วางลิงก์ ตรวจสอบข้อมูลอัตโนมัติ และดาวน์โหลดตามรูปแบบที่ต้องการ",
      to: "/projects/video-downloader",
      tag: isEN ? "Utility" : "เครื่องมือ",
    },
    {
      title: isEN ? "Text Art" : "ภาพจากตัวอักษร",
      desc: isEN
        ? "Convert image to high-detail ASCII text with fine controls and live preview."
        : "แปลงรูปเป็น ASCII แบบละเอียด พร้อมปรับค่าและพรีวิวทันที",
      to: "/projects/text-art",
      tag: isEN ? "Creative" : "สร้างสรรค์",
    },
  ];

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#060f24] font-[Space_Grotesk] text-slate-100">
      <div className="pointer-events-none absolute inset-0">
        <motion.div
          animate={{ x: [0, 20, -16, 0], y: [0, -12, 10, 0] }}
          transition={{ duration: 14, ease: "easeInOut", repeat: Infinity }}
          className="absolute -top-24 left-1/3 h-80 w-80 rounded-full bg-cyan-400/20 blur-[130px]"
        />
        <motion.div
          animate={{ x: [0, -18, 12, 0], y: [0, 14, -12, 0] }}
          transition={{ duration: 16, ease: "easeInOut", repeat: Infinity }}
          className="absolute bottom-0 right-0 h-96 w-96 rounded-full bg-emerald-300/15 blur-[145px]"
        />
      </div>

      <div className="relative mx-auto max-w-6xl px-4 pb-16 pt-24 sm:px-8">
        <motion.header
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: "easeOut" }}
          className="mb-8 rounded-3xl border border-white/10 bg-slate-900/50 p-8 backdrop-blur-xl"
        >
          <p className="text-[11px] uppercase tracking-[0.35em] text-cyan-300/80">
            {isEN ? "Projects" : "โปรเจกต์"}
          </p>
          <h1 className="mt-3 text-4xl font-semibold tracking-tight text-white sm:text-5xl">
            {isEN ? "Project Lab" : "พื้นที่ทดลองและผลงาน"}
          </h1>
          <p className="mt-3 max-w-2xl text-sm text-slate-300">
            {isEN
              ? "Each card opens a standalone mini-app. UI is tuned for desktop and mobile usage."
              : "แต่ละการ์ดคือแอปย่อยแยกกัน พร้อม UI ที่ใช้งานได้ทั้งเดสก์ท็อปและมือถือ"}
          </p>
        </motion.header>

        <section className="grid gap-5 sm:grid-cols-2">
          {cards.map((card, index) => (
            <motion.div
              key={card.to}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay: index * 0.06, ease: "easeOut" }}
            >
              <Link
                to={card.to}
                className="group block rounded-3xl border border-white/10 bg-slate-900/45 p-6 backdrop-blur-xl transition-all hover:-translate-y-1 hover:border-cyan-300/60 hover:bg-slate-900/65"
              >
                <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.25em] text-cyan-200/80">
                  <span>{card.tag}</span>
                  <span className="text-slate-500 group-hover:text-cyan-200">Open</span>
                </div>
                <h2 className="mt-4 text-xl font-semibold text-white">{card.title}</h2>
                <p className="mt-3 text-sm leading-relaxed text-slate-300">{card.desc}</p>
                <div className="mt-5 h-px bg-gradient-to-r from-cyan-400/0 via-cyan-300/45 to-emerald-200/0 opacity-0 transition group-hover:opacity-100" />
              </Link>
            </motion.div>
          ))}
        </section>
      </div>
    </main>
  );
}
