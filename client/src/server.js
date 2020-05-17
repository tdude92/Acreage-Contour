const path = require("path");
const express = require("express");
var fs = require("fs");
var cors = require("cors");
var bodyParser = require("body-parser");

module.exports = {
  app: function () {
    const app = express();
    app.use(bodyParser.urlencoded({ extended: false }));
    app.use(bodyParser.json());
    const indexPath = path.join(__dirname, "index.html");
    const publicPath = express.static(path.join(__dirname, "../dist"));

    app.use("/dist", publicPath);
    app.get("/", function (_, res) {
      res.sendFile(indexPath);
    });
    app.post("/input", cors(), function (req, res) {
      console.log(req);
      res.download(a[p]);
    });
    app.listen(3000, function () {
      console.log("Started on PORT 3000");
    });

    return app;
  },
};
