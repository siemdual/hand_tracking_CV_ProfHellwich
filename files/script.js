const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const hidden = document.getElementById('hidden-canvas');
const depthView = document.getElementById('depth');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById('webcamButton');


let classesDir = {
    1: {
        name: 'palm',
        id: 1,
    },
    2: {
        name: 'palm',
        id: 2,
    }
}



// Check if webcam access is supported.
function getUserMediaSupported() {
  return !!(navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user
// wants to activate it to call enableCam function which we will 
// define in the next step.
if (getUserMediaSupported()) {
  enableWebcamButton.addEventListener('click', enableCam);
} else {
  console.warn('getUserMedia() is not supported by your browser');
}


// Enable the live webcam view and start classification.
function enableCam(event) {
  // Only continue if the COCO-SSD has finished loading.
  if (!model) {
    return;
  }
  
  // Hide the button once clicked.
  event.target.classList.add('removed');  
  
  // getUsermedia parameters to force video but not audio.
  const constraints = {
    video: { width: 440, height: 280 }
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
    video.srcObject = stream;
    video.addEventListener('loadeddata', predictWebcam);
  });
}


//var model = undefined;
model_url = 'https://raw.githubusercontent.com/siemdual/app_palm_detect/main/models/palm-detector/model.json';
//Call load function
const model = asyncLoadModel(model_url);
//console.log(model);

//Function Loads the GraphModel type model of
async function asyncLoadModel(model_url) {
    console.log('Model loaded');
    //Enable start button:
    demosSection.classList.remove('invisible');
    enableWebcamButton.innerHTML = 'Start camera';
    return await tf.loadGraphModel(model_url);
}



var children = [];

function predictWebcam() {
  // Now let's start classifying a frame in the stream.
  //Get next video frame:
    //await tf.nextFrame();
    //Create tensor from image:
  const tfImg = tf.browser.fromPixels(video);

    //Create smaller image which fits the detection size
    //const smallImg = tf.image.resizeBilinear(tfImg, [vidHeight,vidWidth]);


    //const resized = tf.cast(smallImg, 'int32');
  var tf4d_ = tf.tensor4d(Array.from(tfImg.dataSync()), [1,tfImg.shape[0], tfImg.shape[1], 3]);
  const tf4d = tf.cast(tf4d_, 'int32');

  
  tf4d_.dispose();

    //Perform the detection with your layer model:
  model.then( function(model) { 
    const predictions = model.executeAsync(tf4d).then((predictions)=> {

      //console.log(predictions[6].dataSync());

      const bboxes= predictions[6].dataSync() ;
      const classes = predictions[2].dataSync();
      const scores = predictions[3].dataSync() ;

      for (let i = 0; i < children.length; i++) {
        liveView.removeChild(children[i]);
      }
      children.splice(0);


      var detectionObjects = []
        // Now lets loop through predictions and draw them to the live view if
        // they have a high confidence score.
      for (let n = 0; n < classes.length; n++) {

        const minY = bboxes[n*4] * tf4d.shape[1];
        const minX = bboxes[n*4 +1] * tf4d.shape[2];
        const maxY = bboxes[n*4 +2] * tf4d.shape[1];
        const maxX = bboxes[n*4 +3] * tf4d.shape[2];

        // If we are over 66% sure we are sure we classified it right, draw it!
        if (scores[n*3+2 ] > 0.37 && checkOverlap(0.01, minX, minY, maxX, maxY, scores[n*3+2 ], detectionObjects) ) {


          const bbox = [];
          bbox[0] = minX;
          bbox[1] = minY;
          bbox[2] = maxX ;
          bbox[3] = maxY ;
          detectionObjects.push({
            class: classes[n],
            label: classesDir[classes[n]].name,
            score: scores[n*3+2],
            bbox: bbox
          });


          const p = document.createElement('p');
          p.innerText = classesDir[classes[n]].name  + ' - with ' 
            + Math.round(parseFloat(scores[n*3+2]) * 100) 
            + '% confidence.';
          p.style = 'margin-left: ' + minX + 'px; margin-top: '
            + (minY - 10) + 'px; width: ' 
            + (maxX-minX - 10) + 'px; top: 0; left: 0;';
          const highlighter = document.createElement('div');
          highlighter.setAttribute('class', 'highlighter');
          highlighter.style = 'left: ' + minX + 'px; top: '
            + minY + 'px; width: ' 
            + (maxX-minX) + 'px; height: '
            + (maxY-minY) + 'px;';
          liveView.appendChild(highlighter);
          liveView.appendChild(p);
          children.push(highlighter);
          children.push(p);
        }
      }

      //find highscore object for cropping

      var highscoreIndex = -1;
      var highscore =0;
      var winner = undefined;

      detectionObjects.forEach(function(item, index, object) {

        if (item.score> highscore){
          highscore = item.score;
          highscoreIndex = index;
        }
      });

      winner= detectionObjects[highscoreIndex];

      if (winner != undefined){
        var context = hidden.getContext('2d');
        context.drawImage(video, 0, 0, 440, 280);

        let src = cv.imread('hidden-canvas');
        tfImg.dispose();
        let dst = new cv.Mat();
        // You can try more different parameters
        let rect = new cv.Rect(winner.bbox[0], winner.bbox[1], winner.bbox[2]-winner.bbox[0], winner.bbox[3]-winner.bbox[1]);
        dst = src.roi(rect);
        cv.imshow('depth', dst);
        src.delete();
        dst.delete();
      }
      


        // Call this function again to keep predicting when the browser is ready.
      window.requestAnimationFrame(predictWebcam);

    });
      
  });

}



function checkOverlap(threshold, minX, minY, maxX, maxY, score, detectionObjects){

  var removeList =[];
  var check = true;

  detectionObjects.forEach(function(item, index, object) {

    
    //determine the coordinates of the intersection rectangle
    const x_left = Math.max(minX, item.bbox[0]);
    const y_top = Math.max(minY, item.bbox[1]);
    const x_right = Math.min(maxX, item.bbox[2]);   
    const y_bottom = Math.min(maxY, item.bbox[3]);

    if (x_right > x_left && y_bottom > y_top){
      //The intersection of two axis-aligned bounding boxes is always an
      //axis-aligned bounding box
      const intersection_area = (x_right - x_left) * (y_bottom - y_top);
      //compute the area of both AABBs
      const bb1_area = (maxX - minX) * (maxY- minY);
      const bb2_area = (item.bbox[2] - item.bbox[0]) * (item.bbox[3] -item.bbox[1]);
      // compute the intersection over union by taking the intersection
      // area and dividing it by the sum of prediction + ground-truth
      //areas - the interesection area
      const iou = intersection_area / (bb1_area + bb2_area - intersection_area);

      console.log("iou: "+ iou);
    

      if(iou > threshold){

        console.log("found overlap");
        if(score < item.score){
          check = false;
        }else{
          removeList.push(index);
        }
      }
    }
  });

  if (check==true){
    //iterate reverse to prevent messing up indices
    for (var i = removeList.length -1; i >= 0; i--)
      detectionObjects.splice(removeList[i],1);
  }

  return check;

}

