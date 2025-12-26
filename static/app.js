const form = document.getElementById("form");
const btn = document.getElementById("btn");
const statusEl = document.getElementById("status");
const stateEl = document.getElementById("state");
const pctEl = document.getElementById("pct");
const barFill = document.getElementById("barFill");
const resultEl = document.getElementById("result");
const player = document.getElementById("player");
const download = document.getElementById("download");
const errorEl = document.getElementById("error");

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function setProgress(status, progress) {
  stateEl.textContent = status;
  const pct = Math.round((progress ?? 0) * 100);
  pctEl.textContent = `${pct}%`;
  barFill.style.width = `${pct}%`;
}

async function pollJob(jobId) {
  while (true) {
    const res = await fetch(`/jobs/${jobId}`);
    const job = await res.json();

    if (job.error) {
      hide(statusEl);
      errorEl.textContent = job.error;
      show(errorEl);
      btn.disabled = false;
      return;
    }

    setProgress(job.status, job.progress);

    if (job.status === "done" && job.output) {
      const url = `/audio/${job.output}`;
      player.src = url;
      download.href = url;
      download.textContent = `Download ${job.output}`;
      hide(statusEl);
      show(resultEl);
      btn.disabled = false;
      return;
    }

    await new Promise(r => setTimeout(r, 700));
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  hide(resultEl);
  hide(errorEl);
  show(statusEl);
  setProgress("uploading", 0.02);
  btn.disabled = true;

  const fd = new FormData(form);

  const res = await fetch("/start", { method: "POST", body: fd });
  const data = await res.json();

  if (!res.ok) {
    hide(statusEl);
    errorEl.textContent = data?.error ?? "Failed to start job";
    show(errorEl);
    btn.disabled = false;
    return;
  }

  setProgress("queued", 0.05);
  await pollJob(data.job_id);
});
