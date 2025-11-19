let video = document.getElementById("webcam");
let startBtn = document.getElementById("startBtn");
let stopBtn = document.getElementById("stopBtn");
let resultText = document.getElementById("resultText");
let subjectSelect = document.getElementById("subjectSelect");

let scanning = false;
let intervalId = null;

// Start webcam
async function startCamera() {
    let stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
}

// Convert current frame to Blob
function captureFrame() {
    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    let ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    return new Promise((resolve) => {
        canvas.toBlob(resolve, "image/jpeg");
    });
}

// Send frame to backend
async function sendFrame() {
    if (!scanning) return;

    let subject_id = subjectSelect.value;
    let blob = await captureFrame();

    let formData = new FormData();
    formData.append("image", blob, "frame.jpg");
    formData.append("subject_id", subject_id);
    let section = document.getElementById("sectionSelect").value;
    formData.append("section", section);



    try {
        let res = await fetch("http://127.0.0.1:5000/recognize_and_mark", {
            method: "POST",
            body: formData
        });

        let data = await res.json();

        // Update UI
        if (data.result === "present") {
            resultText.textContent = `✔ Present: ${data.student_name} (ID: ${data.student_id})`;
            resultText.style.color = "green";
        }
        else if (data.result === "already_marked") {
            resultText.textContent = `ℹ Already marked: ${data.student_id}`;
            resultText.style.color = "blue";
        }
        else if (data.result === "unknown") {
            resultText.textContent = "❌ Unknown face";
            resultText.style.color = "red";
        }
        else {
            resultText.textContent = "⚠ Error in backend";
            resultText.style.color = "orange";
        }

    } catch (err) {
        resultText.textContent = "Server error / offline";
        resultText.style.color = "orange";
    }
}

// Event: Start scanning
startBtn.onclick = () => {
    scanning = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;

    // send a frame every 1.5 seconds
    intervalId = setInterval(sendFrame, 1500);
};

// Event: Stop scanning
stopBtn.onclick = () => {
    scanning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;

    clearInterval(intervalId);
    resultText.textContent = "Stopped.";
};

// Start webcam on page load
startCamera();
