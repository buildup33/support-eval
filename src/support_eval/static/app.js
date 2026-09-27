(() => {
  const page = document.body.dataset.page;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  if (page === "home") {
    const hero = document.querySelector("[data-ambient-hero]");
    const input = document.querySelector("[data-file-input]");
    const dropZone = document.querySelector("[data-drop-zone]");
    const fileCard = document.querySelector("[data-file-card]");
    const fileName = document.querySelector("[data-file-name]");
    const fileMeta = document.querySelector("[data-file-meta]");
    const fileType = document.querySelector("[data-file-type]");
    const remove = document.querySelector("[data-remove-file]");
    const replace = document.querySelector("[data-replace-file]");
    const editor = document.querySelector("[data-rubric-editor]");
    const preview = document.querySelector("[data-criterion-preview]");
    const status = document.querySelector("[data-rubric-status]");
    const count = document.querySelector("[data-criteria-count]");
    const runCount = document.querySelector("[data-run-criteria]");
    const form = document.querySelector("[data-evaluation-form]");
    const runButton = document.querySelector("[data-run-button]");

    if (hero && !reducedMotion.matches && window.matchMedia("(pointer: fine)").matches) {
      hero.addEventListener("pointermove", (event) => {
        const bounds = hero.getBoundingClientRect();
        hero.style.setProperty("--light-x", `${event.clientX - bounds.left}px`);
        hero.style.setProperty("--light-y", `${event.clientY - bounds.top}px`);
      });
    }

    const showFile = () => {
      const file = input.files[0];
      if (!file) {
        fileCard.hidden = true;
        dropZone.classList.remove("has-file");
        return;
      }
      fileName.textContent = file.name;
      fileType.textContent = file.name.split(".").pop().toUpperCase();
      fileMeta.textContent = `${(file.size / 1024).toFixed(file.size > 1024 ? 0 : 1)} KB · Ready to evaluate`;
      fileCard.hidden = false;
      dropZone.classList.add("has-file");
    };

    input?.addEventListener("change", showFile);
    remove?.addEventListener("click", () => {
      input.value = "";
      showFile();
      input.focus();
    });
    replace?.addEventListener("click", () => input.click());
    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone?.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("is-dragging");
      });
    });
    dropZone?.addEventListener("dragleave", () => dropZone.classList.remove("is-dragging"));
    dropZone?.addEventListener("drop", (event) => {
      event.preventDefault();
      dropZone.classList.remove("is-dragging");
      if (event.dataTransfer?.files.length) {
        input.files = event.dataTransfer.files;
        showFile();
      }
    });

    const renderRubric = () => {
      try {
        const data = JSON.parse(editor.value);
        if (!Array.isArray(data.criteria) || data.criteria.length === 0) {
          throw new Error("Add at least one criterion");
        }
        preview.replaceChildren(
          ...data.criteria.map((criterion) => {
            const card = document.createElement("article");
            card.className = "preview-card";
            const header = document.createElement("header");
            const title = document.createElement("strong");
            title.textContent = criterion.name || criterion.id || "Unnamed criterion";
            const id = document.createElement("code");
            id.textContent = criterion.id || "missing-id";
            const description = document.createElement("p");
            description.textContent = criterion.description || "No description";
            const chips = document.createElement("div");
            const weight = document.createElement("span");
            weight.textContent = `Weight ${criterion.weight ?? "—"}`;
            const flag = document.createElement("span");
            flag.textContent = criterion.flag_on_fail ? "Critical on fail" : "Score only";
            chips.append(weight, flag);
            header.append(title, id);
            card.append(header, description, chips);
            return card;
          }),
        );
        count.textContent = `${data.criteria.length} ${data.criteria.length === 1 ? "criterion" : "criteria"}`;
        runCount.textContent = data.criteria.length;
        status.textContent = `Valid JSON · ${data.name || "Unnamed rubric"}`;
        status.classList.remove("is-invalid");
        editor.removeAttribute("aria-invalid");
      } catch (error) {
        preview.replaceChildren();
        count.textContent = "Invalid rubric";
        runCount.textContent = "—";
        status.textContent =
          error instanceof SyntaxError ? "Fix the JSON syntax before running." : error.message;
        status.classList.add("is-invalid");
        editor.setAttribute("aria-invalid", "true");
      }
    };
    editor?.addEventListener("input", renderRubric);
    renderRubric();
    form?.addEventListener("submit", () => {
      runButton.disabled = true;
      runButton.querySelector("span").textContent = "Starting…";
    });
  }

  if (page === "progress") {
    const root = document.querySelector("[data-job-status-url]");
    const statusUrl = root?.dataset.jobStatusUrl;
    let timer;
    const update = async () => {
      try {
        const response = await fetch(statusUrl, { headers: { Accept: "application/json" } });
        if (!response.ok) throw new Error("Status unavailable");
        const data = await response.json();
        document.querySelector("[data-status-message]").textContent = data.message;
        document.querySelector("[data-progress-count]").textContent =
          `${data.completed} of ${data.total}`;
        document.querySelector("[data-progress-percent]").textContent = `${data.percentage}%`;
        const bar = document.querySelector("[data-progress-bar]");
        bar.setAttribute("aria-valuenow", data.completed);
        document.querySelector("[data-progress-fill]").style.width = `${data.percentage}%`;
        if (data.state === "complete") {
          window.location.assign(data.results_url);
          return;
        }
        if (data.state === "failed") {
          document.querySelector("[data-error-message]").textContent = data.error;
          document.querySelector("[data-failure-card]").hidden = false;
          return;
        }
        timer = window.setTimeout(update, 1200);
      } catch (_) {
        document.querySelector("[data-status-message]").textContent =
          "Reconnecting to local evaluation status…";
        timer = window.setTimeout(update, 2500);
      }
    };
    if (statusUrl) update();
    window.addEventListener("pagehide", () => window.clearTimeout(timer));
  }

  if (page === "results") {
    const buttons = [...document.querySelectorAll("[data-filter]")];
    const select = document.querySelector("[data-criterion-filter]");
    const rows = [...document.querySelectorAll("[data-result-row]")];
    const empty = document.querySelector("[data-filter-empty]");
    let state = "all";
    const apply = () => {
      let visible = 0;
      rows.forEach((row) => {
        const stateMatch = state === "all" || row.dataset.state === state;
        const criterionMatch =
          !select.value || row.dataset.criteria.split(" ").includes(select.value);
        row.hidden = !(stateMatch && criterionMatch);
        if (!row.hidden) visible += 1;
      });
      empty.hidden = visible !== 0;
    };
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        state = button.dataset.filter;
        buttons.forEach((item) => {
          const active = item === button;
          item.classList.toggle("active", active);
          item.setAttribute("aria-pressed", String(active));
        });
        apply();
      });
    });
    select?.addEventListener("change", apply);
  }

  if (page === "conversation") {
    let highlightTimer;
    document.querySelectorAll("[data-evidence-target]").forEach((control) => {
      control.addEventListener("click", () => {
        const target = document.getElementById(control.dataset.evidenceTarget);
        if (!target) return;
        window.clearTimeout(highlightTimer);
        document.querySelectorAll(".message.is-highlighted").forEach((item) => {
          item.classList.remove("is-highlighted");
        });
        document.querySelectorAll(".evidence-link.is-active").forEach((item) => {
          item.classList.remove("is-active");
        });
        control.classList.add("is-active");
        target.classList.add("is-highlighted");
        target.scrollIntoView({
          behavior: reducedMotion.matches ? "auto" : "smooth",
          block: "center",
        });
        target.focus({ preventScroll: true });
        highlightTimer = window.setTimeout(() => {
          target.classList.remove("is-highlighted");
          control.classList.remove("is-active");
        }, reducedMotion.matches ? 900 : 2400);
      });
    });
  }
})();
