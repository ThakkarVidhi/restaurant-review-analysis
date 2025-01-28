document.addEventListener("DOMContentLoaded", () => {
  const socket = io("/progress");
  const mainProgressBarFill = document.getElementById("mainProgressBarFill");
  const subProgressBarFill = document.getElementById("subProgressBarFill");
  const progressPercent = document.getElementById("progressPercent");
  const subProgressPercent = document.getElementById("subProgressPercent");
  const progressMessage = document.getElementById("progressMessage");
  const subProgressMessage = document.getElementById("subProgressMessage");
  const progressContainer = document.getElementById("progress-container");
  const analysisForm = document.getElementById("analysis-form");
  const contentWrapper = document.querySelector(".content-wrapper");

  function applyBlurEffect() {
      contentWrapper.classList.add("blurred");
  }

  function removeBlurEffect() {
      contentWrapper.classList.remove("blurred");
  }

  function disableScroll() {
      document.body.style.overflow = "hidden";
  }

  function enableScroll() {
      document.body.style.overflow = "";
  }

  function updateProgress(mainProgress, mainMessage, subProgress = null, subMessage = null) {
    
    // Update main progress bar
    mainProgressBarFill.style.width = `${mainProgress}%`;
    progressPercent.textContent = `${mainProgress}%`;
    progressMessage.textContent = mainMessage;

    // Update sub progress bar
    if (subProgress !== null && subMessage !== null) {
        subProgressBarFill.style.width = `${subProgress}%`;
        subProgressPercent.textContent = `${subProgress}%`;
        subProgressMessage.textContent = subMessage;
        subProgressMessage.classList.remove("hidden");
        subProgressBarFill.parentElement.classList.remove("hidden");
    } else {
        subProgressMessage.classList.add("hidden");
        subProgressBarFill.parentElement.classList.add("hidden");
    }
  }


  // Show progress bar, blur effect, and disable scrolling on form submission
  analysisForm.addEventListener("submit", (event) => {
      event.preventDefault(); // Prevent default form submission
      const formData = new FormData(analysisForm);

      // Start the fetch request to the backend
      fetch(analysisForm.action, {
          method: "POST",
          body: formData
      })
          .then(response => response.json())
          .then(data => {
              if (data.task_id) {
                  const taskId = data.task_id;

                  // Show the progress container
                  progressContainer.classList.remove("hidden");
                  // Apply blur effect and disable scrolling
                  applyBlurEffect();
                  disableScroll();

                  // Listen for progress updates for this task
                  socket.off(`task_progress_${taskId}`); // Ensure no duplicate listeners
                  socket.on(`task_progress_${taskId}`, (data) => {
                      const { progress, sub_progress, message, sub_message } = data;

                      // Update progress bars dynamically
                      updateProgress(progress, message, sub_progress, sub_message);

                      // If task is complete, redirect to results page
                      if (progress === 100) {
                          setTimeout(() => {
                              removeBlurEffect();
                              enableScroll(); // Re-enable scrolling
                              window.location.href = `/result/${taskId}`;
                          }, 500);
                      }
                  });

                  // Check if socket is connected
                  socket.on('connect', () => {
                      console.log('Socket connected!'); // Log when the socket connects
                  });

                  // Check if socket is disconnected
                  socket.on('disconnect', () => {
                      console.log('Socket disconnected!'); // Log when the socket disconnects
                  });
              } else {
                  console.error("Task ID not received from server.");
              }
          })
          .catch(error => {
              console.error("Error starting analysis task:", error);
          });
  });
});