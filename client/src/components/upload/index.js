import React, { useState, useEffect, useRef } from "react";
import "./Upload.scss";
import PropTypes from "prop-types";
import {
  Paper,
  Container,
  Grid,
  Typography,
  Button,
  Icon,
  Card,
  CardMedia,
  Fade,
  Box,
} from "@material-ui/core";
import { Image } from "@material-ui/icons";
import { motion, AnimatePresence } from "framer-motion";
import Loading from "pages/loading";

function Upload(props) {
  const [dragging, toggleDragging] = useState(false);
  const [fileName, addFileName] = useState(null);
  const [uploadImage, addUploadImage] = useState(null);
  const [uploadFile, addUploadFile] = useState(null);
  const [uploading, toggleUploading] = useState(false);

  const [error, setError] = useState(false);
  const fileInputRef = useRef();
  var dragCounter = 0;

  useEffect(() => {
    try {
      const div = dropRef.current;
      div.addEventListener("dragenter", handleDragIn);
      div.addEventListener("dragleave", handleDragOut);
      div.addEventListener("dragover", handleDrag);
      div.addEventListener("drop", handleDrop);
      return () => {
        // unsubscribe event
        div.removeEventListener("dragenter", handleDragIn);
        div.removeEventListener("dragleave", handleDragOut);
        div.removeEventListener("dragover", handleDrag);
        div.removeEventListener("drop", handleDrop);
      };
    } catch (err) {}
  }, [uploading]);

  const serverURL = "http://localhost:5000/";
  //handle drag and drop visuals
  const dropRef = useRef();
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  const handleDragIn = (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter++;
    console.log(dragCounter);
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      toggleDragging(true);
    }
  };
  const handleDragOut = (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter--;
    if (dragCounter > 0) return;
    toggleDragging(false);
  };
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    toggleDragging(false);
    const acceptedImageTypes = ["image/jpeg"];
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (acceptedImageTypes.includes(file["type"])) {
        addUploadImage(URL.createObjectURL(file));
        addUploadFile(file);
        addFileName(file.name.split(/(\\|\/)/g).pop());
        e.dataTransfer.clearData();
        dragCounter = 0;
      } else {
        setError(true);
      }
    }
  };

  //handle files
  const clearData = () => {
    addFileName(null);
    addUploadImage(null);
    addUploadFile(null);
    setError(null);
  };

  const openFileDialog = () => {
    fileInputRef.current.click();
  };

  const onFileAdded = (event) => {
    event.preventDefault();
    const file = event.target.files[0];
    addUploadFile(file);
    addUploadImage(URL.createObjectURL(file));
    addFileName(event.target.value.split(/(\\|\/)/g).pop());
  };

  const uploadFileToServer = () => {
    toggleUploading(true);
    var myHeaders = new Headers();
    myHeaders.append("Content-Type", "application/json");
    var reader = new FileReader();
    reader.readAsDataURL(uploadFile);
    reader.onloadend = function () {
      var base64data = reader.result;
      var requestOptions = {
        method: "POST",
        headers: myHeaders,
        mode: "cors",
        body: JSON.stringify({ data: base64data }),
      };

      fetch(serverURL, requestOptions)
        .then((response) => response.json())
        .then((data) => {
          console.log(data);

          props.setInput(`data:image/jpeg;base64,${data.input}`);
          props.setOutput(`data:image/jpeg;base64,${data.output}`);
        })
        .then(() => {
          clearData();
          toggleUploading(false);
        })
        .catch((error) => {
          console.log("error", error);
          clearData();
          toggleUploading(false);
        });
    };
  };

  return uploading ? (
    <AnimatePresence>
      <motion.div
        className="load-container"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <Loading />
      </motion.div>
    </AnimatePresence>
  ) : (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <Paper
          style={{
            boxSizing: "content-box",
            border: dragging ? "4px solid #DD2C00" : "4px solid #FFFFFF",
          }}
          ref={dropRef}
          elevation={4}
        >
          <Container>
            <Grid
              style={{ padding: 16 }}
              container
              alignContent="center"
              alignItems="center"
              justify="center"
              direction="column"
            >
              {uploadImage == null ? (
                <React.Fragment>
                  <Image style={{ fontSize: "12em" }} />
                  <Typography
                    style={{ marginBottom: "0.2em", fontSize: "1.5rem" }}
                    align="center"
                  >
                    Drag and drop your image files to upload
                  </Typography>
                  <Typography
                    style={{ marginBottom: "0.2em" }}
                    align="center"
                    color="textSecondary"
                  >
                    Put your jpeg image of the landscape to get an accurate
                    detailed segmented land cover map
                  </Typography>
                  <Button
                    onClick={openFileDialog}
                    style={{ margin: 16 }}
                    variant="contained"
                    color="secondary"
                  >
                    Choose Jpeg Image
                  </Button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    onChange={onFileAdded}
                    accept="image/jpeg"
                    style={{ display: "none" }}
                  />
                  {error ? (
                    <Typography
                      style={{ marginBottom: "0.2em" }}
                      align="center"
                      color="error"
                    >
                      File must be in jpeg format
                    </Typography>
                  ) : null}
                </React.Fragment>
              ) : (
                <React.Fragment>
                  <Typography
                    style={{ marginBottom: "0.2em", fontSize: "2rem" }}
                  >
                    Preview Image
                  </Typography>
                  <Card style={{ maxWidth: "90%" }}>
                    <CardMedia>
                      <img
                        src={uploadImage}
                        style={{ width: "100%", height: "auto" }}
                      />
                      <Button onClick={clearData}>Remove Image</Button>
                    </CardMedia>
                  </Card>
                  <Button
                    style={{ margin: 16 }}
                    variant="contained"
                    color="secondary"
                    onClick={uploadFileToServer}
                  >
                    Upload Image
                  </Button>
                </React.Fragment>
              )}
            </Grid>
          </Container>
        </Paper>
      </motion.div>
    </AnimatePresence>
  );
}
Upload.propTypes = {
  setInput: PropTypes.func,
  setOutput: PropTypes.func,
};

export default Upload;
