const path = require("path");
const express = require("express");
var bodyParser = require("body-parser");
const { spawn } = require("child_process");
const fs = require("fs");

const app = express(),
  DIST_DIR = __dirname,
  HTML_FILE = path.join(DIST_DIR, "index.html");

//post requests
app.use(bodyParser()); // to use bodyParser (for text/number data transfer between client and server)
app.use(express.static(__dirname + "/public")); // making ./public as the static directory
app.set("views", __dirname + "/views"); // making ./views as the views directory
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

//client get requests
app.use(express.static(DIST_DIR));
app.get("*", function (_, res) {
  res.sendFile(indexPath);
});

app.post("/upload", function (req, res) {
  if (req.file) {
    console.log("Uploading file...");
    var filename = req.file.filename;
    var uploadStatus = "File Uploaded Successfully";
  } else {
    console.log("No File Uploaded");
    var filename = "FILE NOT UPLOADED";
    var uploadStatus = "File Upload Failed";
  }
});

//functions
function run_script(file_name) {
  // Spawn a new child process
  spawn("python", ["../ml/app.py", file_name]);
  fs.unlink("outputs/" + file_name);
}

// Define port for app to listen on
const port = process.env.PORT || 3000;
// To make the server live
app.listen(port, () => {
  console.log(`App is live on port ${port}`);
});
