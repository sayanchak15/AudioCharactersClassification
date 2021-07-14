var recordButton = document.getElementById("recordButtonBTN");

//add events to those 2 buttons
recordButton.addEventListener("click", startRecording);

// document.getElementById("text").innerHTML = "";



function startRecording() {
    console.log("recordButton clicked");
    document.getElementById("text").innerHTML = "";
    document.getElementById("word").innerHTML = "";
    recordButton.disabled = true;
    recordButton.innerText = "Recording...."
    console.log(document);
    console.log(recordButton);
}

