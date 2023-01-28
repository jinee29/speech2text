//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream;                      //stream from getUserMedia()
var rec;                            //Recorder.js object
var input;                          //MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb.
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record

var recordButton = document.getElementById("recordButton");

//add events to those 2 buttons
recordButton.addEventListener("click", startRecording);

function startRecording() {
    console.log("recordButton clicked");

    var constraints = { audio: true, video:false }

    recordButton.disabled = true;
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        audioContext = new AudioContext({sampleRate: 8000});

        //update the format
        document.getElementById("formats").innerHTML="Format: 1 channel pcm @ "+audioContext.sampleRate/1000+"kHz"

        /*  assign to gumStream for later use  */
        gumStream = stream;

        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);

        rec = new Recorder(input,{numChannels:1})

        //start the recording process
        rec.record()
        timeout_status  = setTimeout( function(){stopRecording();},
            2000 );
         console.log(timeout_status);
    }).catch(function(err) {
        recordButton.disabled = false;
    });
}


function stopRecording() {
    //disable the stop button, enable the record too allow for new recordings
    recordButton.disabled = false;

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createDownloadLink);

}


function createDownloadLink(blob) {

    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    var link = document.createElement('a');

    //name of .wav file to use during upload and download (without extendion)
    var filename = new Date().toISOString();

    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    //save to disk link
    link.href = url;
    link.download = filename+".wav"; //download forces the browser to donwload the file using the  filename
    link.innerHTML = "Save to disk";

    //add the new audio element to li
    li.appendChild(au);

    //add the filename to the li
    li.appendChild(document.createTextNode(filename+".wav "))

    //add the save to disk link to li
    li.appendChild(link);

    //upload link
    var upload = document.createElement('a');
    upload.href="#";
    upload.innerHTML = "Upload";
    var xhr=new XMLHttpRequest();
    xhr.onload=function(e) {
              if(this.readyState === 4) {
                  console.log("Server returned: ",e.target.responseText);}};
    var fd=new FormData();
    fd.append("audio_data",blob, filename);
    xhr.open("POST","/",true);
    xhr.send(fd);
    li.appendChild(document.createTextNode (" "))//add a space in between

    //add the li element to the ol
    recordingsList.appendChild(li);
}