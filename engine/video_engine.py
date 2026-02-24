"""
Video Engine for VADAR video analysis.

Extends the VADAR execution model from single-image to video:
instead of a single `image`, the LLM-generated program receives
`video_dir`, `frame_count`, `fps`, and video module functions.
"""

import os
import re
import signal
import sys
import json
import linecache
import runpy
import traceback
from typing import Literal

from PIL import Image
from tqdm import tqdm

from engine.engine_utils import (
    Generator,
    get_methods_from_json,
    TimeoutException,
    timeout_handler,
)
from engine.predefined_modules import VideoModulesList
from engine.video_utils import extract_frames, get_video_metadata


class VideoEngine:
    """
    Execution engine for video analysis programs.

    Similar to the image-mode Engine, but provides video-specific
    variables (video_dir, frame_count, fps) and video module functions
    (track, loc_frame, vqa_frame, get_frame, result) in the namespace.
    """

    def __init__(
        self,
        api_json=None,
        api_key_path="./api.key",
        results_folder_path="",
        models_path="",
        dataset="omni3d",
        chunk_size=None,
    ):
        self.api_json = api_json
        self.results_folder_path = results_folder_path

        print("Initializing video modules")
        self.modules_list = VideoModulesList(
            models_path=models_path,
            dataset=dataset,
            api_key_path=api_key_path,
            chunk_size=chunk_size,
        )

        if api_json:
            self.api_methods, self.namespace = get_methods_from_json(self.api_json)
        else:
            self.api_methods = []
            self.namespace = {}

        # Add video module functions to namespace
        self.namespace.update(self.modules_list.module_executes)

        self.trace_file_path = ""
        self.program_executable_path = ""
        self.result_file = ""
        self.execution_json = []
        self.namespace_line = sys.maxsize
        self.api_key_path = api_key_path
        self.output_csv_path = os.path.join(results_folder_path, "outputs.csv")

    def process_video(
        self,
        video_path,
        questions,
        results_folder_path=None,
        frames_fps=None,
    ):
        """
        Process a video file with a list of questions.

        1. Extracts frames from the video.
        2. For each question, runs the LLM-generated program.

        Args:
            video_path (str): Path to the input video file.
            questions (list): List of question dicts with 'question' key.
            results_folder_path (str, optional): Override results path.
            frames_fps (float, optional): FPS for frame extraction.

        Returns:
            list: Execution results for each question.
        """
        if results_folder_path is None:
            results_folder_path = self.results_folder_path

        # Extract frames
        frames_dir = os.path.join(results_folder_path, "extracted_frames")
        print(f"Extracting frames from {video_path}...")
        video_info = extract_frames(video_path, frames_dir, fps=frames_fps)
        print(
            f"Extracted {video_info['frame_count']} frames at "
            f"{video_info['target_fps']:.1f} FPS"
        )

        # Get video metadata for the namespace
        frame_count = video_info["frame_count"]
        fps = video_info["target_fps"]

        return self.execute_programs_on_frames(
            frames_dir, frame_count, fps, questions, results_folder_path
        )

    def execute_programs_on_frames(
        self,
        video_dir,
        frame_count,
        fps,
        questions,
        results_folder_path=None,
    ):
        """
        Execute programs on pre-extracted video frames.

        Args:
            video_dir (str): Path to directory of extracted JPEG frames.
            frame_count (int): Number of frames in the directory.
            fps (float): Frames per second of the video.
            questions (list): List of dicts, each with keys:
                - 'question' (str): The question text
                - 'question_index' (int): Question identifier
                - 'program' (str): LLM-generated Python program
            results_folder_path (str, optional): Override results path.

        Returns:
            list: Execution results for each question.
        """
        if results_folder_path is None:
            results_folder_path = self.results_folder_path

        folder_name = "video_execution"
        exec_results_path = os.path.join(results_folder_path, folder_name)
        os.makedirs(exec_results_path, exist_ok=True)

        for question_data in tqdm(questions, desc="Processing video questions"):
            question_results_path = os.path.join(
                exec_results_path,
                f"question_{question_data['question_index']}/",
            )
            os.makedirs(question_results_path, exist_ok=True)
            exec_env_path = os.path.join(question_results_path, "exec_env/")
            os.makedirs(exec_env_path, exist_ok=True)

            self.trace_file_path = os.path.join(exec_env_path, "trace.html")
            with open(self.trace_file_path, "w+") as f:
                f.write(f"<h1>Question: {question_data['question']}</h1>")

            self.program_executable_path = os.path.join(
                exec_env_path, "executable_program.py"
            )
            self.result_file = os.path.join(exec_env_path, "result.json")

            self.execution_json.append(
                self.run_video_program(
                    question_data,
                    video_dir,
                    frame_count,
                    fps,
                    error_count=0,
                )
            )

        # Save execution results
        execution_json_path = os.path.join(exec_results_path, "execution.json")
        with open(execution_json_path, "w+") as file:
            json.dump(self.execution_json, file, indent=2)

        return self.execution_json

    def run_video_program(
        self,
        question_data,
        video_dir,
        frame_count,
        fps,
        error_count=0,
    ):
        """
        Execute a single video analysis program.

        Args:
            question_data (dict): Must contain 'question' and 'program' keys.
            video_dir (str): Path to extracted frames directory.
            frame_count (int): Total number of frames.
            fps (float): Video FPS.
            error_count (int): Current retry count for error correction.

        Returns:
            dict: Execution result with question, program, and answer.
        """
        program = question_data.get("program", "")

        self.modules_list.set_trace_path(self.trace_file_path)
        execution_data = {}

        # Inject video-specific variables into namespace
        self.namespace.update(
            video_dir=video_dir,
            frame_count=frame_count,
            fps=fps,
        )

        try:
            if isinstance(program, list):
                program = program[0]
        except Exception as e:
            if error_count < 5:
                print("No program found")
                corrected_data = self._correct_program_error(
                    question_data, Exception("No program found")
                )
                return self.run_video_program(
                    corrected_data, video_dir, frame_count, fps, error_count + 1
                )
            else:
                program = ""

        self._add_program_to_file(program)

        error = self._execute_file()
        if error and error_count < 5:
            corrected_data = self._correct_program_error(
                question_data, error
            )

            if os.path.exists(self.trace_file_path):
                os.remove(self.trace_file_path)

            return self.run_video_program(
                corrected_data, video_dir, frame_count, fps, error_count + 1
            )

        # Read results
        try:
            with open(self.result_file, "r") as f:
                result_namespace = json.load(f)
        except Exception as e:
            result_namespace = {"final_result": f"Error: {error}"}

        execution_data["execution"] = {}
        execution_data["execution"]["question"] = question_data
        execution_data["execution"]["program"] = program
        execution_data["execution"]["result_namespace"] = result_namespace

        if "final_result" in result_namespace:
            final_result = result_namespace["final_result"]
            if isinstance(final_result, bool):
                execution_data["execution"]["answer"] = (
                    "yes" if final_result else "no"
                )
            elif isinstance(final_result, str):
                execution_data["execution"]["answer"] = final_result.lower()
            else:
                execution_data["execution"]["answer"] = final_result
        else:
            execution_data["execution"]["answer"] = ""

        return execution_data

    def _correct_program_error(self, question_data, error):
        """Ask the LLM to fix a program that produced an error."""
        messages = question_data.get("messages", [])
        messages.append(
            {
                "role": "user",
                "content": (
                    f"\nThere was an error in running the code: {error}. "
                    "Try again and include the program between "
                    "<program></program>"
                ),
            }
        )
        generator = Generator(
            question_data.get("model_name", "gpt-4o"),
            api_key_path=self.api_key_path,
        )
        output, messages = generator.generate(None, messages)
        output = generator.remove_substring(output, "```python")
        output = generator.remove_substring(output, "```")
        program = re.findall(r"<program>(.*?)</program>", output, re.DOTALL)

        corrected = dict(question_data)
        corrected["program"] = program[0] if program else ""
        corrected["messages"] = messages
        corrected["output"] = output
        return corrected

    def _add_program_to_file(self, program):
        """Write the program to an executable file with namespace serialization."""
        with open(self.program_executable_path, "w") as file:
            file.write("import math\n")
            file.writelines(f"{method}\n" for method in self.api_methods)
            file.write("\n# PROGRAM STARTS HERE\n")

        new_program_content = [f"{line}\n" for line in program.split("\n")]

        write_namespace_code = f"""
# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {{k: v for k, v in globals().items() if is_serializable(v)}}

with open("{self.result_file}", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        """

        with open(self.program_executable_path, "a") as file:
            file.writelines(new_program_content)
            file.write(write_namespace_code)

    def _trace_execution(self, frame, event, arg):
        """Trace function to log program execution."""
        if event == "line":
            filename = frame.f_globals.get("__file__", None)
            if filename and os.path.basename(filename) == os.path.basename(
                self.program_executable_path
            ):
                lineno = frame.f_lineno
                line = linecache.getline(filename, lineno).strip()
                if lineno > self.namespace_line:
                    return self._trace_execution
                if "import math" in line:
                    return self._trace_execution
                if "import" in line:
                    self.namespace_line = lineno
                    return self._trace_execution
                function_name = frame.f_code.co_name
                trace_line = f"<p>{lineno}: "
                if function_name and function_name != "<module>":
                    trace_line += f"[In method {function_name}] "
                trace_line += f"<code>{line}</code></p>\n"
                with open(self.trace_file_path, "a+") as f:
                    f.write(trace_line)
        return self._trace_execution

    def _execute_file(self):
        """Execute the program file with tracing and timeout."""
        sys.settrace(self._trace_execution)
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(300)  # 5 minute timeout for video processing
            runpy.run_path(
                self.program_executable_path, init_globals=self.namespace
            )
            signal.alarm(0)
        except TimeoutException as e:
            return e
        except Exception as e:
            return e
        finally:
            sys.settrace(None)
        return None
