let video = document.getElementById('video');

if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
        video.srcObject = stream;
        video.play();
    });
}

document.getElementById('loginButton').addEventListener('click', function() {
    let canvas = document.createElement('canvas');
    canvas.width = 1920;
    canvas.height = 1080;
    let context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, 1920, 1080);
    canvas.toBlob(function(blob) {
        let formData = new FormData();
        formData.append('image', blob);
        fetch('/authenticate', {
            method: 'POST',
            body: formData
        }).then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.message);
                });
            }
            return response.json();
        }).then(data => {
            if (data.success) {
                alert(data.message);
            }
        }).catch(error => {
            alert(error.message);
        });
    });
});
