import React from "react";
import {
  Paper,
  Container,
  Grid,
  Typography,
  Button,
  Icon,
  Card,
  CardContent,
  CardHeader,
  CardMedia,
} from "@material-ui/core";
import ArrowRightAltIcon from "@material-ui/icons/ArrowRightAlt";

function Result(props) {
  return (
    <Paper
      style={{
        boxSizing: "content-box",
      }}
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
          <Typography variant="h3" style={{ padding: 16 }}>
            Network Result
          </Typography>
          <Card>
            <CardHeader
              title="Input Image"
              style={{ padding: "8px" }}
              titleTypographyProps={{ style: { fontSize: "1.5rem" } }}
            />
            <CardMedia>
              <img
                src={example}
                style={{ minWidth: 512, width: "100%", height: "auto" }}
              />
            </CardMedia>
            <a
              style={{
                color: "black",
                padding: "8px",
                display: "inline-block",
              }}
              href={example}
              download="output"
            >
              Download
            </a>
          </Card>
          <ArrowRightAltIcon
            style={{ fontSize: "4rem", transform: "rotate(90deg)", margin: 8 }}
          />
          <Card>
            <CardHeader
              title="Output Image"
              style={{ padding: "8px" }}
              titleTypographyProps={{ style: { fontSize: "1.5rem" } }}
            />
            <CardMedia>
              <img
                src={example}
                style={{ minWidth: 512, width: "100%", height: "auto" }}
              />
              <a
                style={{
                  color: "black",
                  padding: "8px",
                  display: "inline-block",
                }}
                href={example}
                download="output"
              >
                Download
              </a>
            </CardMedia>
          </Card>
          <Button style={{ margin: 16 }} variant="contained" color="secondary">
            Choose Another Image
          </Button>
        </Grid>
      </Container>
    </Paper>
  );
}

export default Result;
