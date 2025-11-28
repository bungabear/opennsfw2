"""
Prediction router for NSFW detection endpoints.
"""
import time
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, status
from PIL import Image
from pydantic import BaseModel

import opennsfw2 as n2
from ..pydantic_models import (
    ErrorResponse,
    MultipleImagesRequest,
    MultipleImagesResponse,
    PredictionResult,
    SingleImageRequest,
    SingleImageResponse,
    VideoRequest,
    VideoResponse,
    VideoResult,
    SinglePathRequest,
    MultiplePathsRequest,
    VideoPathRequest,
    PathOptions,
)

from ..services.prediction_service import PredictionService
from ..services.file_service import FileService
from ..utils.exceptions import InvalidInputError, DownloadError

router = APIRouter()


@router.post(
    "/image",
    response_model=SingleImageResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict_image(request: SingleImageRequest) -> SingleImageResponse:
    """Predict NSFW probability for a single image."""
    start_time = time.time()

    try:
        service = PredictionService()

        processed_input = FileService.process_input_data(request.input)

        if not isinstance(processed_input, Image.Image):
            raise InvalidInputError("Input is not a valid image.")

        nsfw_prob = service.predict_image(
            processed_input,
            preprocessing=request.options.preprocessing if request.options else n2.Preprocessing.YAHOO
        )

        processing_time = (time.time() - start_time) * 1000

        return SingleImageResponse(
            result=PredictionResult(nsfw_probability=nsfw_prob),
            processing_time_ms=processing_time,
            version=n2.__version__
        )

    except InvalidInputError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e
    except DownloadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Download failed: {e}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}"
        ) from e


@router.post(
    "/images",
    response_model=MultipleImagesResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict_images(request: MultipleImagesRequest) -> MultipleImagesResponse:
    """Predict NSFW probabilities for multiple images."""
    start_time = time.time()

    try:
        service = PredictionService()

        processed_inputs: List[Image.Image] = []
        for input_data in request.inputs:
            processed_input = FileService.process_input_data(input_data)

            if not isinstance(processed_input, Image.Image):
                raise InvalidInputError(f"Input {input_data.data[:50]}... is not a valid image")

            processed_inputs.append(processed_input)

        nsfw_probs = service.predict_images(
            processed_inputs,
            preprocessing=request.options.preprocessing if request.options else n2.Preprocessing.YAHOO
        )

        results = [
            PredictionResult(nsfw_probability=nsfw_prob)
            for nsfw_prob in nsfw_probs
        ]

        processing_time = (time.time() - start_time) * 1000

        return MultipleImagesResponse(
            results=results,
            processing_time_ms=processing_time,
            version=n2.__version__
        )

    except InvalidInputError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e
    except DownloadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Download failed: {e}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}"
        ) from e


@router.post(
    "/video",
    response_model=VideoResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict_video(request: VideoRequest) -> VideoResponse:
    """Predict NSFW probabilities for video frames."""
    start_time = time.time()

    try:
        service = PredictionService()

        # Process video input and get temporary file.
        with FileService.process_video_input(request.input) as video_path:
            elapsed_seconds, nsfw_probabilities = service.predict_video(
                video_path,
                preprocessing=request.options.preprocessing if request.options else n2.Preprocessing.YAHOO,
                frame_interval=request.options.frame_interval if request.options else 8,
                aggregation_size=request.options.aggregation_size if request.options else 8,
                aggregation=request.options.aggregation if request.options else n2.Aggregation.MEAN
            )

        processing_time = (time.time() - start_time) * 1000

        return VideoResponse(
            result=VideoResult(
                elapsed_seconds=elapsed_seconds,
                nsfw_probabilities=nsfw_probabilities
            ),
            processing_time_ms=processing_time,
            version=n2.__version__
        )

    except InvalidInputError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e
    except DownloadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Download failed: {e}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}"
        ) from e

def _resolve_preprocessing(options: Optional[PathOptions]) -> n2.Preprocessing:
    if not options or options.preprocessing is None:
        return n2.Preprocessing.YAHOO
    val = options.preprocessing
    if isinstance(val, n2.Preprocessing):
        return val
    if isinstance(val, str):
        try:
            return getattr(n2.Preprocessing, val)
        except Exception:
            try:
                return n2.Preprocessing[val.upper()]
            except Exception:
                return n2.Preprocessing.YAHOO
    return n2.Preprocessing.YAHOO


def _resolve_aggregation(options: Optional[PathOptions]) -> n2.Aggregation:
    if not options or options.aggregation is None:
        return n2.Aggregation.MEAN
    val = options.aggregation
    if isinstance(val, n2.Aggregation):
        return val
    if isinstance(val, str):
        try:
            return getattr(n2.Aggregation, val)
        except Exception:
            try:
                return n2.Aggregation[val.upper()]
            except Exception:
                return n2.Aggregation.MEAN
    return n2.Aggregation.MEAN


@router.post(
    "/image/path",
    response_model=SingleImageResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict_image_by_path(request: SinglePathRequest) -> SingleImageResponse:
    """Predict NSFW probability for a single image by local file path (no HTTP upload)."""
    start_time = time.time()
    try:
        service = PredictionService()

        try:
            img = Image.open(request.path)
            # ensure image is loaded
            img.load()
        except FileNotFoundError as e:
            raise InvalidInputError(f"File not found: {request.path}") from e
        except OSError as e:
            raise InvalidInputError(f"Cannot open image file: {request.path}") from e

        preprocessing = _resolve_preprocessing(request.options)

        nsfw_prob = service.predict_image(
            img,
            preprocessing=preprocessing
        )

        processing_time = (time.time() - start_time) * 1000

        return SingleImageResponse(
            result=PredictionResult(nsfw_probability=nsfw_prob),
            processing_time_ms=processing_time,
            version=n2.__version__
        )

    except InvalidInputError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}"
        ) from e


@router.post(
    "/images/paths",
    response_model=MultipleImagesResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict_images_by_paths(request: MultiplePathsRequest) -> MultipleImagesResponse:
    """Predict NSFW probabilities for multiple images by local file paths (no HTTP upload)."""
    start_time = time.time()
    try:
        service = PredictionService()

        imgs: List[Image.Image] = []
        for p in request.paths:
            try:
                img = Image.open(p)
                img.load()
            except FileNotFoundError as e:
                raise InvalidInputError(f"File not found: {p}") from e
            except OSError as e:
                raise InvalidInputError(f"Cannot open image file: {p}") from e
            imgs.append(img)

        preprocessing = _resolve_preprocessing(request.options)

        nsfw_probs = service.predict_images(
            imgs,
            preprocessing=preprocessing
        )

        results = [PredictionResult(nsfw_probability=prob) for prob in nsfw_probs]

        processing_time = (time.time() - start_time) * 1000

        return MultipleImagesResponse(
            results=results,
            processing_time_ms=processing_time,
            version=n2.__version__
        )

    except InvalidInputError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}"
        ) from e


@router.post(
    "/video/path",
    response_model=VideoResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict_video_by_path(request: VideoPathRequest) -> VideoResponse:
    """Predict NSFW probabilities for a local video file path (no HTTP upload)."""
    start_time = time.time()
    try:
        service = PredictionService()

        # Validate path exists by letting predict_video use it, but check common errors first
        try:
            # just ensure file is accessible
            with open(request.path, "rb"):
                pass
        except FileNotFoundError as e:
            raise InvalidInputError(f"File not found: {request.path}") from e
        except OSError as e:
            raise InvalidInputError(f"Cannot access video file: {request.path}") from e

        preprocessing = _resolve_preprocessing(request.options)
        frame_interval = request.options.frame_interval if request.options and request.options.frame_interval is not None else 8
        aggregation_size = request.options.aggregation_size if request.options and request.options.aggregation_size is not None else 8
        aggregation = _resolve_aggregation(request.options)

        elapsed_seconds, nsfw_probabilities = service.predict_video(
            request.path,
            preprocessing=preprocessing,
            frame_interval=frame_interval,
            aggregation_size=aggregation_size,
            aggregation=aggregation
        )

        processing_time = (time.time() - start_time) * 1000

        return VideoResponse(
            result=VideoResult(
                elapsed_seconds=elapsed_seconds,
                nsfw_probabilities=nsfw_probabilities
            ),
            processing_time_ms=processing_time,
            version=n2.__version__
        )

    except InvalidInputError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}"
        ) from e
