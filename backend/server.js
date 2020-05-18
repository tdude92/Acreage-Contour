/* =================================================== */
/* ===== Section 1: Require all the dependencies ===== */
/* =================================================== */

const express = require('express');
const fileUpload = require('express-fileupload');
const bodyParser = require('body-parser');
const hbs = require('hbs');
const logger = require('morgan');
const {spawn} = require('child_process');
const fs = require("fs");

// Define port for app to listen on
const port =  process.env.PORT || 8080;

/* ==================================================== */
/* ===== Section 2: Configure express middlewares ===== */
/* ==================================================== */

const app =  express();
app.use(fileUpload());
app.use(bodyParser());  // to use bodyParser (for text/number data transfer between clientg and server)
app.set('view engine', 'hbs');  // setting hbs as the view engine
app.use(express.static(__dirname + '/public'));  // making ./public as the static directory
app.set('views', __dirname + '/views');  // making ./views as the views directory
app.use(logger('dev'));  // Creating a logger (using morgan)
app.use(express.json());
app.use(express.urlencoded({ extended: false }));


/* ==================================== */
/* ===== Section 3: Making Routes ===== */
/* ==================================== */

// GET / route for serving index.html file
app.get('/', (req, res) => {
    res.render('index.hbs');
});

// POST /upload for file upload
/* ===== Make sure that file name matches the name attribute in your html ===== */
app.post('/upload', function(req, res) {
    if (!req.files || Object.keys(req.files).length === 0) {
      return res.status(400).send('No files were uploaded.');
    }
  
    // The name of the input field (i.e. "sampleFile") is used to retrieve the uploaded file
    //let sampleFile = req.files.sampleFile;
    let  file = req.files.sampleFile;


    // Use the mv() method to place the file somewhere on your server
    sampleFile.mv('/upload/test.jpg', function(err) {
      if (err)
        return res.status(500).send(err);
  
      res.send('File uploaded!');
    });
  });



// GET /temp to render temp.hbs, for dev purposes
app.get('/temp', (req, res) => {
    res.render('temp.hbs');
});

// To make the server live
app.listen(port, () => {
    console.log(`App is live on port ${port}`);
});


