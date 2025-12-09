# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


import cv2
import numpy as np

from ultralytics.utils import LOGGER


class GMC:
    """
    Generalized Motion Compensation (GMC) class for tracking and object detection in video frames.

    This class provides methods for tracking and detecting objects based on several tracking algorithms including ORB,
    SIFT, ECC, and Sparse Optical Flow. It also supports downscaling of frames for computational efficiency.

    Attributes:
        method (str): The tracking method to use. Options include 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'.
        downscale (int): Factor by which to downscale the frames for processing.
        prevFrame (np.ndarray): Previous frame for tracking.
        prevKeyPoints (list): Keypoints from the previous frame.
        prevDescriptors (np.ndarray): Descriptors from the previous frame.
        initializedFirstFrame (bool): Flag indicating if the first frame has been processed.

    Methods:
        apply: Apply the chosen method to a raw frame and optionally use provided detections.
        apply_ecc: Apply the ECC algorithm to a raw frame.
        apply_features: Apply feature-based methods like ORB or SIFT to a raw frame.
        apply_sparseoptflow: Apply the Sparse Optical Flow method to a raw frame.
        reset_params: Reset the internal parameters of the GMC object.

    Examples:
        Create a GMC object and apply it to a frame
        >>> gmc = GMC(method="sparseOptFlow", downscale=2)
        >>> frame = np.array([[1, 2, 3], [4, 5, 6]])
        >>> processed_frame = gmc.apply(frame)
        >>> print(processed_frame)
        array([[1, 2, 3],
               [4, 5, 6]])
    """

    def __init__(self, method: str = "sparseOptFlow", downscale: int = 2) -> None:
        """
        Initialize a Generalized Motion Compensation (GMC) object with tracking method and downscale factor.

        Args:
            method (str): The tracking method to use. Options include 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'.
            downscale (int): Downscale factor for processing frames.

        Examples:
            Initialize a GMC object with the 'sparseOptFlow' method and a downscale factor of 2
            >>> gmc = GMC(method="sparseOptFlow", downscale=2)
        """
        super().__init__()

        self.method = method
        self.downscale = max(1, downscale)

        if self.method == "orb":
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        elif self.method == "sift":
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        elif self.method == "ecc":
            number_of_iterations = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        elif self.method == "sparseOptFlow":
            self.feature_params = dict(
                maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04
            )

        elif self.method in {"none", "None", None}:
            self.method = None
        else:
            raise ValueError(f"Error: Unknown GMC method: {method}")

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False

    def apply(self, raw_frame: np.ndarray, detections: list = None) -> np.ndarray:
        """
        Apply object detection on a raw frame using the specified method.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed, with shape (H, W, C).
            detections (List | None): List of detections to be used in the processing.

        Returns:
            (np.ndarray): Transformation matrix with shape (2, 3).

        Examples:
            >>> gmc = GMC(method="sparseOptFlow")
            >>> raw_frame = np.random.rand(480, 640, 3)
            >>> transformation_matrix = gmc.apply(raw_frame)
            >>> print(transformation_matrix.shape)
            (2, 3)
        """
        if self.method in {"orb", "sift"}:
            return self.apply_features(raw_frame, detections)
        elif self.method == "ecc":
            return self.apply_ecc(raw_frame)
        elif self.method == "sparseOptFlow":
            return self.apply_sparseoptflow(raw_frame)
        else:
            return np.eye(2, 3)

    def apply_ecc(self, raw_frame: np.ndarray) -> np.ndarray:
        """
        Apply the ECC (Enhanced Correlation Coefficient) algorithm to a raw frame for motion compensation.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed, with shape (H, W, C).

        Returns:
            (np.ndarray): Transformation matrix with shape (2, 3).

        Examples:
            >>> gmc = GMC(method="ecc")
            >>> processed_frame = gmc.apply_ecc(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
            >>> print(processed_frame)
            [[1. 0. 0.]
             [0. 1. 0.]]
        """
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        # Downscale image
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            dsheight, dswidth = height // self.downscale, width // self.downscale
            frame = cv2.resize(frame, (dswidth, dsheight))
        # Handle first frame

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Run the ECC algorithm. The results are stored in warp_matrix.
        # (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria)
        try:
            (_, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except Exception as e:
            LOGGER.warning(f"WARNING: find transform failed. Set warp as identity {e}")

        return H

    def apply_features(self, raw_frame: np.ndarray, detections: list = None) -> np.ndarray:
        """
        Apply feature-based methods like ORB or SIFT to a raw frame.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed, with shape (H, W, C).
            detections (List | None): List of detections to be used in the processing.

        Returns:
            (np.ndarray): Transformation matrix with shape (2, 3).

        Examples:
            >>> gmc = GMC(method="orb")
            >>> raw_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> transformation_matrix = gmc.apply_features(raw_frame)
            >>> print(transformation_matrix.shape)
            (2, 3)
        """
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image
        if self.downscale > 1.0:
            dswidth = width // self.downscale
            dsheight = height // self.downscale
            frame = cv2.resize(frame, (dswidth, dsheight))
            width, height = dswidth, dsheight

        # Mask preparation (optimized)
        mask = self._prepare_mask(frame, width, height, detections, self.downscale)

        # Feature detection and extraction (no changes due to OpenCV calls, but avoid redundant operations)

        keypoints = self.detector.detect(frame, mask)

        # Compute the descriptors
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            # Use list instead of copy.copy for simple and likely faster copy (keypoints is a list already)
            self.prevKeyPoints = keypoints[:]
            # For large arrays, np.copy is preferable for performance over copy.copy
            self.prevDescriptors = np.copy(descriptors) if descriptors is not None else None

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Match descriptors
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)

        # Filter matches based on smallest spatial distance
        matches = []
        spatialDistances = []

        maxSpatialDistance = 0.25 * np.array([width, height])

        # Handle empty matches case
        if len(knnMatches) == 0:
            # Store to next iteration
            self.prevFrame = frame.copy()
            self.prevKeyPoints = keypoints[:]
            self.prevDescriptors = np.copy(descriptors) if descriptors is not None else None

            return H

        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prev_pt = self.prevKeyPoints[m.queryIdx].pt
                curr_pt = keypoints[m.trainIdx].pt

                dx = prev_pt[0] - curr_pt[0]
                dy = prev_pt[1] - curr_pt[1]

                if abs(dx) < maxSpatialDistance[0] and abs(dy) < maxSpatialDistance[1]:
                    spatialDistances.append((dx, dy))
                    matches.append(m)

        if len(spatialDistances) == 0:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = keypoints[:]
            self.prevDescriptors = np.copy(descriptors) if descriptors is not None else None
            return H

        # Use numpy array directly for mean/std/bool indexing
        spatialDistances_np = np.array(spatialDistances)
        meanSpatialDistances = np.mean(spatialDistances_np, axis=0)
        stdSpatialDistances = np.std(spatialDistances_np, axis=0)
        inliers = (spatialDistances_np - meanSpatialDistances) < (2.5 * stdSpatialDistances)

        # Use boolean indexing for goodMatches/points (optimized)
        good_inliers_idx = np.where(np.logical_and(inliers[:, 0], inliers[:, 1]))[0]
        goodMatches = [matches[i] for i in good_inliers_idx]
        prevPoints = np.array([self.prevKeyPoints[matches[i].queryIdx].pt for i in good_inliers_idx])
        currPoints = np.array([keypoints[matches[i].trainIdx].pt for i in good_inliers_idx])

        # Find rigid matrix if enough good matches

        # Draw the keypoint matches on the output image
        # if False:
        #     import matplotlib.pyplot as plt
        #     matches_img = np.hstack((self.prevFrame, frame))
        #     matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
        #     W = self.prevFrame.shape[1]
        #     for m in goodMatches:
        #         prev_pt = np.array(self.prevKeyPoints[m.queryIdx].pt, dtype=np.int_)
        #         curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
        #         curr_pt[0] += W
        #         color = np.random.randint(0, 255, 3)
        #         color = (int(color[0]), int(color[1]), int(color[2]))
        #
        #         matches_img = cv2.line(matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
        #         matches_img = cv2.circle(matches_img, prev_pt, 2, tuple(color), -1)
        #         matches_img = cv2.circle(matches_img, curr_pt, 2, tuple(color), -1)
        #
        #     plt.figure()
        #     plt.imshow(matches_img)
        #     plt.show()

        # Find rigid matrix
        if prevPoints.shape[0] > 4:
            H_res, inliers = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            if H_res is not None:
                H = H_res
                # Handle downscale
                if self.downscale > 1.0:
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale
        else:
            LOGGER.warning("WARNING: not enough matching points")

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = keypoints[:]
        self.prevDescriptors = np.copy(descriptors) if descriptors is not None else None

        return H

    def apply_sparseoptflow(self, raw_frame: np.ndarray) -> np.ndarray:
        """
        Apply Sparse Optical Flow method to a raw frame.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed, with shape (H, W, C).

        Returns:
            (np.ndarray): Transformation matrix with shape (2, 3).

        Examples:
            >>> gmc = GMC()
            >>> result = gmc.apply_sparseoptflow(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
            >>> print(result)
            [[1. 0. 0.]
             [0. 1. 0.]]
        """
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image
        if self.downscale > 1.0:
            dswidth = width // self.downscale
            dsheight = height // self.downscale
            frame = cv2.resize(frame, (dswidth, dsheight))

        # Find the keypoints
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # Handle first frame
        if not self.initializedFirstFrame or self.prevKeyPoints is None:
            self.prevFrame = frame.copy()
            # keypoints is ndarray or None
            self.prevKeyPoints = keypoints.copy() if keypoints is not None else None
            self.initializedFirstFrame = True
            return H

        # Find correspondences
        matchedKeypoints, status, _ = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)
        if status is None or matchedKeypoints is None or self.prevKeyPoints is None:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = keypoints.copy() if keypoints is not None else None
            return H

        # Fast boolean indexing for good correspondences
        status = status.reshape(-1)
        idxs = np.where(status)[0]
        if idxs.size > 0:
            prevPoints = np.array([self.prevKeyPoints[i] for i in idxs])
            currPoints = matchedKeypoints[idxs]
        else:
            prevPoints = np.empty((0, 1, 2))
            currPoints = np.empty((0, 1, 2))

        # Find rigid matrix
        if (prevPoints.shape[0] > 4) and (prevPoints.shape[0] == currPoints.shape[0]):
            H_res, _ = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            if H_res is not None:
                H = H_res
                if self.downscale > 1.0:
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale
        else:
            LOGGER.warning("WARNING: not enough matching points")

        self.prevFrame = frame.copy()
        self.prevKeyPoints = keypoints.copy() if keypoints is not None else None

        return H

    def reset_params(self) -> None:
        """Reset the internal parameters including previous frame, keypoints, and descriptors."""
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False

    def _prepare_mask(self, frame: np.ndarray, width: int, height: int, detections: list, downscale: int) -> np.ndarray:
        # Helper for mask preparation, optimized for bulk assignment
        mask = np.zeros_like(frame)
        y0, y1 = int(0.02 * height), int(0.98 * height)
        x0, x1 = int(0.02 * width), int(0.98 * width)
        mask[y0:y1, x0:x1] = 255
        if detections is not None and len(detections) > 0:
            # Vectorized zeroing
            dets = np.empty((len(detections), 4), dtype=np.int32)
            # Convert all detection boxes at once, exploiting np arrays
            det_arr = np.array([det[:4] for det in detections])
            np.floor_divide(det_arr, downscale, out=dets)
            for tlbr in dets:
                mask[tlbr[1] : tlbr[3], tlbr[0] : tlbr[2]] = 0
        return mask
