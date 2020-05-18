import React from "react";
import "./Loading.scss";
import { motion, AnimatePresence } from "framer-motion";

const Loading = () => (
  <AnimatePresence exitBeforeEnter>
    <motion.div
      className="loading-screen"
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="loading-svg">
        <svg
          width="200"
          height="200"
          viewBox="0 0 669 669"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <motion.path
            d="M559 331.001C559 383.056 540.951 433.5 507.927 473.739C474.904 513.978 428.95 541.522 377.895 551.678C326.841 561.833 273.844 553.972 227.936 529.433C182.027 504.895 146.048 465.197 126.127 417.105C106.207 369.012 103.578 315.5 118.688 265.687C133.799 215.873 165.715 172.84 208.997 143.92C252.279 115 304.25 101.982 356.054 107.084C407.858 112.187 456.291 135.093 493.099 171.902L459.903 205.098C430.775 175.969 392.448 157.842 351.452 153.804C310.457 149.767 269.33 160.069 235.079 182.955C200.827 205.841 175.571 239.895 163.613 279.315C151.655 318.734 153.736 361.081 169.5 399.139C185.264 437.197 213.737 468.612 250.066 488.03C286.396 507.449 328.334 513.67 368.737 505.633C409.139 497.597 445.504 475.8 471.637 443.957C497.77 412.114 512.054 372.195 512.054 331.001L559 331.001Z"
            fill="white"
            animate={{
              scale: [1, 2, 2, 1, 1],
              rotate: [0, 0, 360, 360, 360],
            }}
            transition={{
              duration: 2,
              ease: "easeInOut",
              times: [0, 0.2, 0.5, 0.8, 1],
              loop: Infinity,
              repeatDelay: 1,
            }}
          />
        </svg>
      </div>
    </motion.div>
  </AnimatePresence>
);

export default Loading;
