# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy

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
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

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
            dscale = self.downscale
            width = width // dscale
            height = height // dscale
            frame = cv2.resize(frame, (width, height))
        else:
            dscale = 1.0

        # Allocate mask buffer only once per function

        # Find the keypoints
        mask = np.zeros_like(frame)

        # Fill mask (inner region used for features)
        y0 = int(0.02 * height)
        y1 = int(0.98 * height)
        x0 = int(0.02 * width)
        x1 = int(0.98 * width)
        mask[y0:y1, x0:x1] = 255

        # Zero out detection regions
        if detections is not None and len(detections) > 0:
            detsArr = np.asarray(detections)
            # Vectorized int coords for detection bounding boxes
            tlbrs = (detsArr[:, :4] / dscale).astype(np.int_)
            for tlbr in tlbrs:
                mask[tlbr[1] : tlbr[3], tlbr[0] : tlbr[2]] = 0

        keypoints = self.detector.detect(frame, mask)

        # Compute the descriptors
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Match descriptors
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)

        # Filter matches based on smallest spatial distance
        matches = []
        spatialDistances = []

        maxSpatialDistanceX = 0.25 * width
        maxSpatialDistanceY = 0.25 * height

        # Handle empty matches case
        if not knnMatches:
            # Store to next iteration
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            return H

        # Loops are near unavoidable here, but can avoid tuple creation and unneeded abs, allocate lists once.
        append_spatialDistances = spatialDistances.append
        append_matches = matches.append
        prevKeyPoints = self.prevKeyPoints
        currKeyPoints = keypoints

        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                pkp = prevKeyPoints[m.queryIdx].pt
                ckp = currKeyPoints[m.trainIdx].pt
                dx = pkp[0] - ckp[0]
                dy = pkp[1] - ckp[1]

                # Avoid creating tuple and use fast abs primitives
                if abs(dx) < maxSpatialDistanceX and abs(dy) < maxSpatialDistanceY:
                    append_spatialDistances((dx, dy))
                    append_matches(m)

        # Early exit if not enough matches
        if not spatialDistances:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        # Numpy calculations for mean/std/inliers
        spatialDistancesArr = np.array(spatialDistances)
        meanSpatialDistances = np.mean(spatialDistancesArr, axis=0)
        stdSpatialDistances = np.std(spatialDistancesArr, axis=0)
        # Broadcasting, avoid creating new array
        inliersMask = (spatialDistancesArr - meanSpatialDistances) < (2.5 * stdSpatialDistances)

        # Collect good matches and their locations
        goodMatches, prevPoints, currPoints = [], [], []

        matchesArr = matches  # length == spatialDistancesArr.shape[0]
        nMatches = spatialDistancesArr.shape[0]
        inliersMask0 = inliersMask[:, 0]
        inliersMask1 = inliersMask[:, 1]

        for i in range(nMatches):
            if inliersMask0[i] and inliersMask1[i]:
                m = matchesArr[i]
                goodMatches.append(m)
                prevPoints.append(prevKeyPoints[m.queryIdx].pt)
                currPoints.append(currKeyPoints[m.trainIdx].pt)

        prevPointsArr = np.array(prevPoints)
        currPointsArr = np.array(currPoints)

        # Find rigid matrix
        if prevPointsArr.shape[0] > 4:
            H_fit, _ = cv2.estimateAffinePartial2D(prevPointsArr, currPointsArr, cv2.RANSAC)
            # Handle downscale
            if self.downscale > 1.0 and H_fit is not None:
                H_fit[0, 2] *= self.downscale
                H_fit[1, 2] *= self.downscale
            if H_fit is not None:
                H = H_fit
        else:
            LOGGER.warning("WARNING: not enough matching points")

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

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
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

            # Find the keypoints
            width = width // self.downscale
            height = height // self.downscale

        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # Handle first frame
        if not self.initializedFirstFrame or self.prevKeyPoints is None:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.initializedFirstFrame = True
            return H

        # Find correspondences
        matchedKeypoints, status, _ = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)

        # Use numpy filtering for speed over list comprehensions for large arrays
        if status is not None and matchedKeypoints is not None and self.prevKeyPoints is not None:
            statusFlat = status.ravel().astype(bool)
            prevPts = np.asarray(self.prevKeyPoints)[statusFlat]
            currPts = np.asarray(matchedKeypoints)[statusFlat]
        else:
            prevPts = np.array([])
            currPts = np.array([])

        # Find rigid matrix if enough matches
        if prevPts.shape[0] > 4 and prevPts.shape[0] == currPts.shape[0]:
            H_fit, _ = cv2.estimateAffinePartial2D(prevPts, currPts, cv2.RANSAC)
            if self.downscale > 1.0 and H_fit is not None:
                H_fit[0, 2] *= self.downscale
                H_fit[1, 2] *= self.downscale
            if H_fit is not None:
                H = H_fit
        else:
            LOGGER.warning("WARNING: not enough matching points")

        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        return H

    def reset_params(self) -> None:
        """Reset the internal parameters including previous frame, keypoints, and descriptors."""
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False
