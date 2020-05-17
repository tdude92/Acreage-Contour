const {spawn} = require('child_process');
const fs = require("fs");

function run_script(file_name) {
    // Spawn a new child process
    spawn('python', ['../ml/app.py', file_name]);
    fs.unlink("outputs/" + file_name);
}