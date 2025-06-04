document.addEventListener("DOMContentLoaded", () => {
  const imageUpload = document.getElementById("imageUpload");
  const uploadButton = document.getElementById("uploadButton");
  const originalImage = document.getElementById("originalImage");
  const processedImage = document.getElementById("processedImage");
  const faceCanvas = document.getElementById("faceCanvas");
  const ctx = faceCanvas.getContext("2d");

  const brightnessSlider = document.getElementById("brightness");
  const brightnessValueSpan = document.getElementById("brightnessValue");
  const contrastSlider = document.getElementById("contrast");
  const contrastValueSpan = document.getElementById("contrastValue");
  const grayscaleCheckbox = document.getElementById("grayscale");
  const gaussianBlurSlider = document.getElementById("gaussianBlur");
  const gaussianBlurValueSpan = document.getElementById("gaussianBlurValue");

  const applyFiltersButton = document.getElementById("applyFiltersButton");
  const detectFacesButton = document.getElementById("detectFacesButton");
  const downloadButton = document.getElementById("downloadButton");

  let uploadedFile = null;
  let currentProcessedImageUrl = null; // Menyimpan URL gambar yang terakhir diproses

  // Update nilai slider di UI
  brightnessSlider.addEventListener("input", () => {
    brightnessValueSpan.textContent = brightnessSlider.value;
  });
  contrastSlider.addEventListener("input", () => {
    contrastValueSpan.textContent = contrastSlider.value;
  });
  gaussianBlurSlider.addEventListener("input", () => {
    gaussianBlurValueSpan.textContent = gaussianBlurSlider.value;
  });

  // Fungsi untuk mengunggah foto
  uploadButton.addEventListener("click", async () => {
    if (!imageUpload.files.length) {
      alert("Pilih berkas gambar terlebih dahulu.");
      return;
    }

    uploadedFile = imageUpload.files[0];
    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (response.ok) {
        alert(data.message);
        const reader = new FileReader();
        reader.onload = (e) => {
          originalImage.src = e.target.result;
          originalImage.style.display = "block";
          processedImage.src = e.target.result; // Awalnya, hasil sama dengan asli
          processedImage.style.display = "block";
          currentProcessedImageUrl = e.target.result; // Untuk deteksi awal atau tanpa filter
          downloadButton.style.display = "block";
          clearCanvas();
        };
        reader.readAsDataURL(uploadedFile);
      } else {
        alert("Error: " + data.error);
      }
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("Terjadi kesalahan saat mengunggah gambar.");
    }
  });

  // Fungsi untuk menerapkan filter
  applyFiltersButton.addEventListener("click", async () => {
    if (!uploadedFile) {
      alert("Unggah foto terlebih dahulu.");
      return;
    }

    const params = {
      brightness: parseFloat(brightnessSlider.value),
      contrast: parseFloat(contrastSlider.value),
      grayscale: grayscaleCheckbox.checked,
      gaussian_blur_radius: parseInt(gaussianBlurSlider.value),
      // Tambahkan parameter lain di sini
    };

    try {
      const response = await fetch("/process_image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(params),
      });
      const data = await response.json();

      if (response.ok) {
        processedImage.src = data.processed_image_url;
        currentProcessedImageUrl = data.processed_image_url; // Update URL gambar yang diproses
        clearCanvas(); // Hapus kotak deteksi wajah sebelumnya
      } else {
        alert("Error: " + data.error);
      }
    } catch (error) {
      console.error("Error processing image:", error);
      alert("Terjadi kesalahan saat memproses gambar.");
    }
  });

  // Fungsi untuk deteksi wajah
  detectFacesButton.addEventListener("click", async () => {
    if (!currentProcessedImageUrl) {
      alert("Proses gambar terlebih dahulu atau unggah foto.");
      return;
    }

    try {
      const response = await fetch("/detect_faces", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // Tidak perlu kirim gambar, backend akan memproses gambar yang terakhir diupload/diproses
        body: JSON.stringify({}),
      });
      const data = await response.json();

      if (response.ok) {
        clearCanvas();
        const faces = data.faces;
        if (faces.length > 0) {
          drawFaces(faces);
        } else {
          alert("Tidak ada wajah terdeteksi.");
        }
      } else {
        alert("Error: " + data.error);
      }
    } catch (error) {
      console.error("Error detecting faces:", error);
      alert("Terjadi kesalahan saat mendeteksi wajah.");
    }
  });

  // Fungsi untuk menggambar kotak wajah
  function drawFaces(faces) {
    // Pastikan kanvas memiliki ukuran yang sama dengan gambar
    faceCanvas.width = processedImage.naturalWidth;
    faceCanvas.height = processedImage.naturalHeight;
    faceCanvas.style.width = processedImage.offsetWidth + "px";
    faceCanvas.style.height = processedImage.offsetHeight + "px";

    ctx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;

    faces.forEach((face) => {
      const { x, y, w, h } = face;
      ctx.strokeRect(x, y, w, h);
    });
  }

  // Fungsi untuk membersihkan kanvas
  function clearCanvas() {
    ctx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
    faceCanvas.width = 0; // Reset ukuran kanvas
    faceCanvas.height = 0;
  }

  // Fungsi untuk mengunduh gambar
  downloadButton.addEventListener("click", () => {
    if (currentProcessedImageUrl) {
      // currentProcessedImageUrl adalah URL relatif dari /static/processed/
      // Kita perlu mengambil nama filenya
      const filename = currentProcessedImageUrl.split("/").pop();
      window.location.href = `/download_image/${filename}`;
    } else {
      alert("Tidak ada gambar untuk diunduh.");
    }
  });

  // History for undo/redo
  let history = [];
  let currentIndex = -1;

  // Add event listeners for new controls
  document.querySelectorAll('input[type="range"]').forEach((slider) => {
    slider.addEventListener("input", updateImage);
  });

  document.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", handleButtonClick);
  });

  function handleButtonClick(e) {
    const operation = e.target.textContent.toLowerCase();
    // Handle different operations
    switch (operation) {
      case "undo":
        undo();
        break;
      case "redo":
        redo();
        break;
      case "reset":
        reset();
        break;
      // Add other operations
    }
  }

  // Add other necessary functions
});
