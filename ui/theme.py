"""Sistema de diseño: tokens y estilos globales inyectados en Streamlit.

Los selectores `data-testid` de Streamlit se han verificado con Streamlit 1.66;
si se actualiza Streamlit conviene revisar esta hoja de estilos.
"""

from __future__ import annotations

import streamlit as st

from core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB
from ui.icons import icon_css, svg_data_uri

_UPLOAD_ICON = svg_data_uri("upload", color="#635BFF", stroke=1.75)

_CSS = """
<style>
:root {
  --od-bg: #F7F8FA;
  --od-surface: #FFFFFF;
  --od-surface-muted: #F3F4F6;
  --od-ink: #17191C;
  --od-ink-2: #6B6F78;          /* #737780 oscurecido para cumplir AA sobre --od-bg */
  --od-ink-3: #9A9EA6;
  --od-border: #E5E7EB;
  --od-border-strong: #D3D6DC;
  --od-accent: #635BFF;
  --od-accent-hover: #5249E6;
  --od-accent-press: #463DD4;
  --od-accent-soft: #F0EFFF;
  --od-accent-ink: #4F46E5;
  --od-focus: 0 0 0 3px rgba(99, 91, 255, .35);
  --od-success: #16A36A; --od-success-ink: #0F7A4F; --od-success-soft: #EAF7F1;
  --od-warning: #E6A23C; --od-warning-ink: #8A5300; --od-warning-soft: #FDF5E8;
  --od-danger:  #DC4545; --od-danger-ink:  #B42323; --od-danger-soft:  #FDEDED;
  --od-r-sm: 8px; --od-r-md: 10px; --od-r-lg: 14px;
  --od-shadow-sm: 0 1px 2px rgba(23, 25, 28, .04);
  --od-shadow-md: 0 1px 2px rgba(23, 25, 28, .04), 0 4px 16px rgba(23, 25, 28, .05);
  --od-mono: ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace;
  --od-ease: cubic-bezier(.2, .7, .2, 1);
}

/* ---------- Base ---------- */
.stApp { background: var(--od-bg); }
[data-testid="stHeader"] { background: var(--od-bg); }
.block-container, [data-testid="stMainBlockContainer"] {
  max-width: 1280px; padding-top: 4rem; padding-bottom: 4rem;
}
.stHtml { font-size: 14px; line-height: 1.5; color: var(--od-ink); }
.stHtml h1, .stHtml h2, .stHtml h3, .stHtml h4 { font-family: inherit; margin: 0; padding: 0; color: var(--od-ink); }
.stHtml p { margin: 0; }
.od-sr-only {
  position: absolute !important; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0;
}
:focus-visible { outline: none; box-shadow: var(--od-focus) !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"][aria-expanded="true"], [data-testid="stSidebar"][data-collapsed="false"] {
  width: 264px !important; min-width: 264px !important;
}
[data-testid="stSidebar"] { border-right: 1px solid var(--od-border); }
[data-testid="stSidebarContent"] { padding-left: 0 !important; padding-right: 0 !important; }
[data-testid="stSidebarUserContent"] { padding: 0 16px 24px !important; }
[data-testid="stSidebarHeader"] { padding: 12px 12px 0 !important; }
.od-brand { display: flex; align-items: center; gap: 12px; padding: 0 4px 20px; margin-top: -8px; }
.od-brand__mark {
  width: 36px; height: 36px; border-radius: var(--od-r-md); flex: none;
  display: grid; place-items: center; color: #fff; background: var(--od-accent);
}
.od-brand__name { font-weight: 650; font-size: 15px; letter-spacing: -.01em; line-height: 1.2; }
.od-brand__sub { font-size: 12px; color: var(--od-ink-2); }
.od-side-label {
  font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  color: var(--od-ink-2); padding: 0 8px; margin: 4px 0 6px;
}
[data-testid="stSidebar"] [data-testid="stPageLink"] a {
  border-radius: var(--od-r-sm); padding: 8px 10px; min-height: 40px; transition: background-color .15s var(--od-ease);
}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover { background: var(--od-surface-muted); }
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"],
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {
  background: var(--od-accent-soft);
}
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] p,
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] span { color: var(--od-accent-ink); font-weight: 600; }
.od-side-models { display: grid; gap: 4px; margin-top: 4px; }
.od-side-model { padding: 8px; border-radius: var(--od-r-sm); }
.od-side-model__role { font-size: 12px; color: var(--od-ink-2); display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.od-side-model__name { font-size: 13px; font-weight: 500; font-family: var(--od-mono); word-break: break-all; margin-top: 2px; }
.od-side-foot { margin-top: 20px; padding: 12px 8px 0; border-top: 1px solid var(--od-border); font-size: 12px; color: var(--od-ink-2); display: grid; gap: 2px; }

/* ---------- Status ---------- */
.od-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; display: inline-block; }
.od-dot.is-live, .od-dot.is-local { background: var(--od-success); box-shadow: 0 0 0 3px var(--od-success-soft); }
.od-dot.is-fallback, .od-dot.is-warning { background: var(--od-warning); box-shadow: 0 0 0 3px var(--od-warning-soft); }
.od-dot.is-unavailable, .od-dot.is-error { background: var(--od-danger); box-shadow: 0 0 0 3px var(--od-danger-soft); }
.od-dot.is-unknown { background: var(--od-ink-3); box-shadow: 0 0 0 3px var(--od-surface-muted); }
.od-state { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 500; white-space: nowrap; }
.od-state.is-live, .od-state.is-local { color: var(--od-success-ink); }
.od-state.is-fallback, .od-state.is-warning { color: var(--od-warning-ink); }
.od-state.is-unavailable, .od-state.is-error { color: var(--od-danger-ink); }
.od-state.is-unknown { color: var(--od-ink-2); }

/* ---------- Header ---------- */
.od-header { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; flex-wrap: wrap; padding-bottom: 8px; }
.od-header h1 { font-size: 28px; line-height: 1.15; font-weight: 700; letter-spacing: -.025em; }
.od-header__sub { font-size: 15px; color: var(--od-ink-2); margin-top: 4px; }
.od-header__tech { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.od-chip {
  display: inline-flex; align-items: center; gap: 6px; height: 26px; padding: 0 10px;
  font-size: 12px; font-weight: 500; color: var(--od-ink-2); background: var(--od-surface);
  border: 1px solid var(--od-border); border-radius: 999px; white-space: nowrap;
}
.od-chip .od-ico { color: var(--od-ink-3); }
.od-header__status { display: flex; flex-wrap: wrap; gap: 8px; }
.od-pill {
  display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px 6px 10px;
  background: var(--od-surface); border: 1px solid var(--od-border); border-radius: 999px; font-size: 12.5px;
}
.od-pill__label { color: var(--od-ink-2); }

/* ---------- Cards ---------- */
[class*="st-key-card-"] {
  background: var(--od-surface); border: 1px solid var(--od-border); border-radius: var(--od-r-lg);
  padding: 20px; box-shadow: var(--od-shadow-sm); gap: 16px;
}
.od-card-head { display: flex; align-items: flex-start; gap: 12px; }
.od-card-head__text { flex: 1; min-width: 0; }
.od-card-head h2 { font-size: 16px; font-weight: 600; letter-spacing: -.01em; line-height: 1.3; }
.od-card-head p { font-size: 13px; color: var(--od-ink-2); margin-top: 2px; }
.od-step {
  width: 26px; height: 26px; border-radius: var(--od-r-sm); flex: none; display: grid; place-items: center;
  font-size: 12px; font-weight: 600; color: var(--od-accent-ink); background: var(--od-accent-soft);
}
.od-step.is-done { color: #fff; background: var(--od-success); }
.od-badge {
  display: inline-flex; align-items: center; gap: 6px; height: 26px; padding: 0 10px; flex: none;
  border-radius: 999px; font-size: 12.5px; font-weight: 600; background: var(--od-accent-soft); color: var(--od-accent-ink);
}
.od-section-title { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin: 16px 0 4px; flex-wrap: wrap; }
.od-section-title h2 { font-size: 18px; font-weight: 650; letter-spacing: -.015em; }
.od-section-title p { font-size: 13px; color: var(--od-ink-2); }

/* ---------- Uploader ---------- */
.st-key-uploader [data-testid="stFileUploaderDropzone"] {
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px;
  min-height: 232px; padding: 28px 20px; cursor: pointer; text-align: center;
  background: #FAFAFC; border: 1.5px dashed var(--od-border-strong); border-radius: var(--od-r-lg);
  transition: border-color .18s var(--od-ease), background-color .18s var(--od-ease);
}
.st-key-uploader [data-testid="stFileUploaderDropzone"]::before {
  content: ""; width: 48px; height: 48px; border-radius: 12px; flex: none;
  background: var(--od-accent-soft) url("__UPLOAD_ICON__") center / 22px no-repeat;
  transition: transform .2s var(--od-ease);
}
.st-key-uploader [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--od-accent); background: #F7F6FF; }
.st-key-uploader [data-testid="stFileUploaderDropzone"]:hover::before { transform: translateY(-2px); }
.st-key-uploader [data-testid="stFileUploaderDropzoneInstructions"] { order: 1; }
.st-key-uploader [data-testid="stFileUploaderDropzone"] > span { order: 2; }
.st-key-uploader [data-testid="stFileUploaderDropzoneInstructions"] span { display: none; }
.st-key-uploader [data-testid="stFileUploaderDropzoneInstructions"] > div::before {
  content: "Arrastra y suelta una imagen aquí"; display: block;
  font-size: 15px; font-weight: 600; color: var(--od-ink); letter-spacing: -.01em;
}
.st-key-uploader [data-testid="stFileUploaderDropzoneInstructions"] > div::after {
  content: "__FORMATS__ · hasta __MAX_MB__ MB"; display: block; margin-top: 4px; font-size: 13px; color: var(--od-ink-2);
}
.st-key-uploader [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {
  min-height: 40px; padding: 0 16px; border-radius: var(--od-r-md); font-weight: 600;
  background: var(--od-surface); border: 1px solid var(--od-border-strong); color: var(--od-ink);
}
.st-key-uploader [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"]:hover { border-color: var(--od-accent); color: var(--od-accent-ink); }
.st-key-uploader [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"] p { display: none; }
.st-key-uploader [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"]::after { content: "Seleccionar archivo"; font-size: 14px; }
/* Con archivo cargado, la zona se compacta para dejar protagonismo a la vista previa. */
.st-key-uploader [data-testid="stFileUploaderDropzone"]:has([data-testid="stFileChips"]) {
  min-height: 0; padding: 8px 10px; flex-direction: row; justify-content: space-between; gap: 8px;
  background: var(--od-surface); border-style: solid; border-color: var(--od-border); text-align: left;
}
.st-key-uploader [data-testid="stFileUploaderDropzone"]:has([data-testid="stFileChips"])::before { display: none; }
.st-key-uploader [data-testid="stFileChips"] { flex: 1; min-width: 0; display: flex; flex-direction: row; align-items: center; justify-content: space-between; gap: 8px; }
.st-key-uploader [data-testid="stFileChips"] > div { min-width: 0; flex: 1; }
.st-key-uploader [data-testid="stFileChips"] > button {
  flex: none; height: 32px; padding: 0 12px; border: 1px solid var(--od-border-strong); border-radius: var(--od-r-sm);
  background: var(--od-surface); color: var(--od-ink); font-size: 13px; font-weight: 600; white-space: nowrap;
}
.st-key-uploader [data-testid="stFileChips"] > button:hover { border-color: var(--od-accent); color: var(--od-accent-ink); }
.st-key-uploader [data-testid="stFileChips"] > button > div { display: none; }
.st-key-uploader [data-testid="stFileChips"] > button::after { content: "Cambiar imagen"; }

/* ---------- Preview & meta ---------- */
.od-preview {
  position: relative; border-radius: var(--od-r-md); overflow: hidden; background: var(--od-surface-muted);
  border: 1px solid var(--od-border); display: grid; place-items: center;
}
.od-preview img { display: block; max-width: 100%; max-height: 340px; width: auto; height: auto; object-fit: contain; }
.od-meta { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
.od-meta__item { background: var(--od-bg); border: 1px solid var(--od-border); border-radius: var(--od-r-sm); padding: 8px 10px; min-width: 0; }
.od-meta__k { display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--od-ink-2); }
.od-meta__v { font-size: 13px; font-weight: 600; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-variant-numeric: tabular-nums; }
.od-meta__item.is-name { grid-column: 1 / -1; }

/* ---------- Buttons ---------- */
.stButton button, .stDownloadButton button {
  min-height: 40px; border-radius: var(--od-r-md); font-weight: 600;
  transition: background-color .15s var(--od-ease), border-color .15s var(--od-ease), color .15s var(--od-ease), transform .1s var(--od-ease);
}
.stButton button:active:not(:disabled), .stDownloadButton button:active:not(:disabled) { transform: translateY(1px); }
.st-key-analyze button { min-height: 52px; font-size: 15px; }
.stButton button[kind="primary"] { background: var(--od-accent); border-color: var(--od-accent); color: #fff; }
.stButton button[kind="primary"]:hover:not(:disabled) { background: var(--od-accent-hover) !important; border-color: var(--od-accent-hover) !important; color: #fff !important; }
.stButton button[kind="primary"]:active:not(:disabled) { background: var(--od-accent-press) !important; border-color: var(--od-accent-press) !important; }
.stButton button:disabled, .stDownloadButton button:disabled {
  background: var(--od-surface-muted) !important; border-color: var(--od-border) !important; color: var(--od-ink-3) !important; cursor: not-allowed;
}
.stButton button[kind="secondary"], .stDownloadButton button[kind="secondary"] { background: var(--od-surface); border-color: var(--od-border-strong); color: var(--od-ink); }
.stButton button[kind="secondary"]:hover:not(:disabled), .stDownloadButton button[kind="secondary"]:hover:not(:disabled) { border-color: var(--od-accent) !important; color: var(--od-accent-ink) !important; background: var(--od-surface) !important; }
.od-hint { font-size: 12.5px; color: var(--od-ink-2); display: flex; align-items: flex-start; gap: 6px; }
.od-hint code { font-family: var(--od-mono); font-size: 12px; color: var(--od-ink); background: var(--od-surface-muted); padding: 1px 5px; border-radius: 4px; }
.od-hint .od-ico { margin-top: 2px; }

/* ---------- Empty / notice states ---------- */
.od-empty {
  display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
  min-height: 380px; padding: 32px 24px; border: 1.5px dashed var(--od-border); border-radius: var(--od-r-md);
  background: repeating-linear-gradient(0deg, transparent 0 23px, rgba(23,25,28,.025) 23px 24px),
              repeating-linear-gradient(90deg, transparent 0 23px, rgba(23,25,28,.025) 23px 24px), var(--od-bg);
}
.od-empty__icon { width: 52px; height: 52px; border-radius: 14px; display: grid; place-items: center; background: var(--od-surface); border: 1px solid var(--od-border); color: var(--od-ink-2); box-shadow: var(--od-shadow-sm); }
.od-empty h3 { font-size: 16px; font-weight: 600; margin-top: 16px; letter-spacing: -.01em; }
.od-empty p { font-size: 13.5px; color: var(--od-ink-2); max-width: 360px; margin-top: 6px; }
.od-empty.is-ready .od-empty__icon { color: var(--od-accent); background: var(--od-accent-soft); border-color: transparent; }
.od-flow { display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 6px; margin: 20px 0 0 !important; list-style: none; padding: 0 !important; }
.od-flow li { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--od-ink-2); margin: 0; }
.od-flow li b { width: 20px; height: 20px; border-radius: 6px; display: inline-grid; place-items: center; font-size: 11px; background: var(--od-surface); border: 1px solid var(--od-border); color: var(--od-ink); }
.od-flow li.is-current { color: var(--od-ink); font-weight: 500; }
.od-flow li.is-current b { background: var(--od-accent); border-color: var(--od-accent); color: #fff; }
.od-flow li.is-done b { background: var(--od-success); border-color: var(--od-success); color: #fff; }
.od-flow .od-flow__sep { color: var(--od-ink-3); }

.od-notice { display: flex; gap: 12px; align-items: flex-start; padding: 14px 16px; border-radius: var(--od-r-md); border: 1px solid; font-size: 13.5px; }
.od-notice > .od-ico { margin-top: 1px; }
.od-notice__title { font-weight: 600; }
.od-notice__body { color: var(--od-ink); margin-top: 2px; }
.od-notice__body code { font-family: var(--od-mono); font-size: 12px; background: rgba(23,25,28,.06); padding: 1px 5px; border-radius: 4px; }
.od-notice details { margin-top: 8px; font-size: 12px; color: var(--od-ink-2); }
.od-notice summary { cursor: pointer; }
.od-notice summary:focus-visible { border-radius: 4px; }
.od-notice.is-error { background: var(--od-danger-soft); border-color: #F5C9C9; }
.od-notice.is-error > .od-ico, .od-notice.is-error .od-notice__title { color: var(--od-danger-ink); }
.od-notice.is-warning { background: var(--od-warning-soft); border-color: #F3DDB5; }
.od-notice.is-warning > .od-ico, .od-notice.is-warning .od-notice__title { color: var(--od-warning-ink); }
.od-notice.is-info { background: var(--od-accent-soft); border-color: #DAD7FF; }
.od-notice.is-info > .od-ico, .od-notice.is-info .od-notice__title { color: var(--od-accent-ink); }
.od-notice.is-success { background: var(--od-success-soft); border-color: #C7EAD9; }
.od-notice.is-success > .od-ico, .od-notice.is-success .od-notice__title { color: var(--od-success-ink); }

/* st.status: integrado en la tarjeta */
[data-testid="stExpander"] details { border-radius: var(--od-r-md); border-color: var(--od-border); background: var(--od-surface); }
[data-testid="stExpander"] summary { font-weight: 500; }
[data-testid="stExpanderDetails"] p, [data-testid="stExpanderDetails"] li { font-size: 13.5px; color: var(--od-ink-2); }
[data-testid="stExpanderDetails"] code { font-size: 12px; color: var(--od-ink); background: var(--od-surface-muted); }

/* ---------- Processing ---------- */
.od-processing { position: relative; }
.od-processing img { filter: saturate(.6) brightness(.97); }
.od-processing::after {
  content: ""; position: absolute; left: 0; right: 0; top: 0; height: 30%;
  background: linear-gradient(180deg, transparent, rgba(99, 91, 255, .16) 85%, rgba(99, 91, 255, .55) 100%);
  border-bottom: 2px solid rgba(99, 91, 255, .7);
  animation: od-scan 1.8s var(--od-ease) infinite alternate;
}
@keyframes od-scan { from { transform: translateY(-100%); } to { transform: translateY(240%); } }

/* ---------- Viewer ---------- */
.od-viewer { margin: 0; }
.od-stage {
  position: relative; width: 100%; max-height: 72vh; margin: 0 auto; overflow: hidden;
  border-radius: var(--od-r-md); background: #0F1012; border: 1px solid var(--od-border);
}
.od-stage__inner { position: relative; margin: 0 auto; max-width: 100%; max-height: 72vh; }
.od-stage img { display: block; width: 100%; height: 100%; object-fit: contain; }
.od-box {
  position: absolute; border: 2px solid var(--c); border-radius: 3px;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, .45), inset 0 0 0 1px rgba(0, 0, 0, .25);
  transition: opacity .18s var(--od-ease), border-width .12s var(--od-ease), box-shadow .18s var(--od-ease);
}
.od-tag {
  position: absolute; left: -2px; bottom: 100%; margin-bottom: 2px; max-width: 220px;
  padding: 2px 7px; border-radius: 5px; background: var(--c); color: var(--t);
  font-size: 11.5px; font-weight: 600; line-height: 1.45; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  box-shadow: 0 1px 2px rgba(0, 0, 0, .35); pointer-events: none; font-variant-numeric: tabular-nums;
}
.od-box.pos-inside .od-tag { bottom: auto; top: 0; left: 0; margin: 0; border-radius: 0 0 5px 0; }
.od-box.pos-below .od-tag { bottom: auto; top: 100%; margin: 2px 0 0; }
.od-box.pos-inside-bottom .od-tag { bottom: 0; left: 0; margin: 0; border-radius: 0 5px 0 0; }
.od-box.is-right .od-tag { left: auto; right: -2px; }
.od-box.is-right.pos-inside .od-tag { right: 0; border-radius: 0 0 0 5px; }
.od-box.is-right.pos-inside-bottom .od-tag { right: 0; border-radius: 5px 0 0 0; }
.od-stage.is-compact .od-tag__full { display: none; }
.od-stage:has(.od-box:hover) .od-box:not(:hover) { opacity: .3; }
.od-box:hover { z-index: 20; border-width: 3px; box-shadow: 0 0 0 1px rgba(0,0,0,.5), 0 0 0 9999px rgba(10, 11, 13, .35); }
.od-box:hover .od-tag__full { display: inline; }
.od-legend-hint { margin-top: 10px; }

/* ---------- Summary ---------- */
.od-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.od-stat { background: var(--od-surface); border: 1px solid var(--od-border); border-radius: var(--od-r-lg); padding: 16px 18px; box-shadow: var(--od-shadow-sm); }
.od-stat__k { display: flex; align-items: center; gap: 8px; font-size: 12.5px; color: var(--od-ink-2); font-weight: 500; }
.od-stat__k .od-ico { color: var(--od-ink-3); }
.od-stat__v { font-size: 28px; font-weight: 650; letter-spacing: -.02em; margin-top: 6px; font-variant-numeric: tabular-nums; line-height: 1.1; }
.od-stat__d { font-size: 12px; color: var(--od-ink-2); margin-top: 4px; }
.od-classes { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
.od-class { display: inline-flex; align-items: center; gap: 6px; height: 28px; padding: 0 10px 0 8px; border: 1px solid var(--od-border); background: var(--od-surface); border-radius: 999px; font-size: 12.5px; }
.od-class i { width: 10px; height: 10px; border-radius: 3px; background: var(--c); box-shadow: inset 0 0 0 1px rgba(0,0,0,.15); }
.od-class b { font-weight: 600; font-variant-numeric: tabular-nums; color: var(--od-ink-2); }

/* ---------- Detection list ---------- */
.od-det-list { list-style: none; margin: 14px 0 0 !important; padding: 0 !important; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.od-det-row {
  margin: 0 !important; display: grid; grid-template-columns: auto minmax(0, 1fr) minmax(96px, 150px); align-items: center; gap: 12px;
  padding: 12px 14px; margin: 0; background: var(--od-surface); border: 1px solid var(--od-border); border-left: 3px solid var(--c);
  border-radius: var(--od-r-md); cursor: default; outline: none;
  transition: border-color .15s var(--od-ease), background-color .15s var(--od-ease), box-shadow .15s var(--od-ease);
}
.od-det-row:hover { background: #FBFBFD; border-color: var(--od-border-strong); border-left-color: var(--c); }
.od-det-row:focus-visible { box-shadow: var(--od-focus) !important; }
.od-det-id { min-width: 34px; height: 24px; padding: 0 6px; border-radius: 6px; display: grid; place-items: center; font-size: 12px; font-weight: 650; background: var(--c); color: var(--t); font-variant-numeric: tabular-nums; }
.od-det-es::first-letter { text-transform: uppercase; }
.od-det-es { font-size: 14px; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.od-det-en { font-size: 12px; color: var(--od-ink-2); margin-top: 1px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.od-lang { font-size: 10px; font-weight: 600; letter-spacing: .04em; color: var(--od-ink-2); border: 1px solid var(--od-border); border-radius: 4px; padding: 0 4px; margin-right: 4px; }
.od-det-box { font-family: var(--od-mono); font-size: 11px; color: var(--od-ink-2); margin-top: 3px; }
.od-conf__v { font-size: 13.5px; font-weight: 650; text-align: right; font-variant-numeric: tabular-nums; }
.od-conf__bar { height: 6px; border-radius: 99px; background: var(--od-surface-muted); margin-top: 6px; overflow: hidden; }
.od-conf__bar span { display: block; height: 100%; border-radius: inherit; background: var(--c); box-shadow: inset 0 0 0 1px rgba(0,0,0,.08); }

/* ---------- Tech info ---------- */
.od-tech { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0; border: 1px solid var(--od-border); border-radius: var(--od-r-md); overflow: hidden; background: var(--od-surface); }
.od-tech__item { padding: 12px 14px; border-right: 1px solid var(--od-border); border-bottom: 1px solid var(--od-border); margin-right: -1px; margin-bottom: -1px; }
.od-tech__k { font-size: 12px; color: var(--od-ink-2); display: flex; align-items: center; gap: 6px; }
.od-tech__v { font-size: 13.5px; font-weight: 600; margin-top: 2px; }
.od-tech__v a { color: var(--od-accent-ink); text-decoration: none; }
.od-tech__v a:hover { text-decoration: underline; }
.od-tech__v code { font-family: var(--od-mono); font-size: 12px; font-weight: 500; color: var(--od-ink-2); }

/* ---------- Models page ---------- */
.od-pipeline { display: flex; align-items: stretch; gap: 8px; flex-wrap: wrap; }
.od-pipeline__step { flex: 1 1 150px; background: var(--od-surface); border: 1px solid var(--od-border); border-radius: var(--od-r-md); padding: 12px 14px; }
.od-pipeline__step small { font-size: 11px; color: var(--od-ink-2); text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }
.od-pipeline__step div { font-size: 13.5px; font-weight: 600; margin-top: 2px; }
.od-pipeline__arrow { display: grid; place-items: center; color: var(--od-ink-3); }
.od-model-card { background: var(--od-surface); border: 1px solid var(--od-border); border-radius: var(--od-r-lg); padding: 18px 20px; display: grid; gap: 12px; height: 100%; }
.od-model-card__head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
.od-model-card__icon { width: 36px; height: 36px; border-radius: var(--od-r-md); display: grid; place-items: center; background: var(--od-accent-soft); color: var(--od-accent-ink); flex: none; }
.od-model-card h3 { font-size: 15px; font-weight: 600; margin-top: 10px; }
.od-model-card p { font-size: 13.5px; color: var(--od-ink-2); }
.od-kv { display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; font-size: 13px; margin: 0 !important; padding: 0 !important; }
.od-kv dt { color: var(--od-ink-2); }
.od-kv dd { margin: 0; font-weight: 500; word-break: break-word; }
.od-kv code { font-family: var(--od-mono); font-size: 12px; }

/* ---------- Motion ---------- */
.od-appear { animation: od-appear .35s var(--od-ease) both; }
.od-det-row { animation: od-appear .3s var(--od-ease) both; animation-delay: calc(var(--i, 0) * 25ms); }
@keyframes od-appear { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
  .od-processing::after { display: none; }
}

/* ---------- Responsive ---------- */
@media (max-width: 1100px) {
  .od-det-list { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 920px) {
  .st-key-workspace > [data-testid="stHorizontalBlock"] { flex-direction: column; gap: 16px; }
  .st-key-workspace > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; min-width: 0; }
  .od-empty { min-height: 260px; }
}
@media (max-width: 640px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 4rem 1rem 3rem; }
  .od-header h1 { font-size: 24px; }
  [class*="st-key-card-"] { padding: 16px; border-radius: 12px; }
  .st-key-uploader [data-testid="stFileUploaderDropzone"] { min-height: 180px; }
  .od-stats { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
  .od-stat { padding: 12px; }
  .od-stat__v { font-size: 22px; }
  .od-stat__d { display: none; }
  .od-det-row { grid-template-columns: auto minmax(0, 1fr); }
  .od-det-row .od-conf { grid-column: 1 / -1; }
  .od-conf__v { text-align: left; }
  .od-stage, .od-stage__inner { max-height: none; }
  /* En pantallas pequeñas solo el identificador; el detalle está en el listado. */
  .od-tag__full { display: none; }
  .od-stage__inner { width: 100% !important; }
  .od-meta { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
"""


def inject_theme() -> None:
    formats = ", ".join(ext.upper() for ext in ALLOWED_EXTENSIONS)
    css = (
        _CSS.replace("__UPLOAD_ICON__", _UPLOAD_ICON)
        .replace("__FORMATS__", formats)
        .replace("__MAX_MB__", str(MAX_UPLOAD_MB))
    )
    st.html(css.replace("</style>", icon_css() + "</style>"))
