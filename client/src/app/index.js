import React, { Suspense } from "react";
import {
  BrowserRouter as Router,
  Route,
  Switch,
  Redirect,
} from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import "./App.scss";
import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";

//pages
const Home = React.lazy(() => {
  return Promise.all([
    import("pages/home"),
    new Promise((resolve) => setTimeout(resolve, 1600)),
  ]).then(([moduleExports]) => moduleExports);
});
const PageNotFound = React.lazy(() => {
  return Promise.all([
    import("pages/pagenotfound"),
    new Promise((resolve) => setTimeout(resolve, 1600)),
  ]).then(([moduleExports]) => moduleExports);
});
import Nav from "components/nav";
import Loading from "pages/loading";

const theme = createMuiTheme({
  palette: {
    primary: {
      main: "#64DD17",
      contrastText: "#000",
    },
    secondary: {
      main: "#DD2C00",
      contrastText: "#fff",
    },
  },
});

//function component
function App() {
  return (
    <Router>
      <ThemeProvider theme={theme}>
        <Suspense className="main-content" fallback={<Loading />}>
          <Nav />
          <Switch>
            <Route exact path="/" component={Home} />
            <Route path="/404" component={PageNotFound} />
            <Redirect to="/404" />
          </Switch>
        </Suspense>
      </ThemeProvider>
    </Router>
  );
}

export default App;
