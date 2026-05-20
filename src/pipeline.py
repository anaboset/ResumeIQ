"""
pipeline.py
-----------
Orchestrates the full ResumeIQ pipeline:
  File Input → Preprocess → Extract → Score → Rank → Output

This is the main entry point for programmatic use.
"""

import logging
from pathlib import Path
from typing import Union

from src.preprocessor import preprocess, preprocess_file
from src.skill_extractor import full_extraction
from src.scorer import ResumeScorer, ScoringConfig
from src.ranker import rank_candidates, summary_report

logger = logging.getLogger(__name__)


def process_text(text: str) -> dict:
    """Run preprocessing + extraction on raw text."""
    prep = preprocess(text)
    extraction = full_extraction(text)
    return {**prep, **extraction}


def process_file(file_path: Union[str, Path]) -> dict:
    """Run preprocessing + extraction on a file."""
    prep = preprocess_file(file_path)
    extraction = full_extraction(prep["raw"])
    result = {**prep, **extraction}
    result["source"] = str(file_path)
    result["name"] = Path(file_path).stem
    return result


class ResumeIQPipeline:
    """
    Full end-to-end pipeline for resume screening.

    Example usage:
        pipeline = ResumeIQPipeline()
        results_df, summary = pipeline.run(
            resumes=["path/to/resume1.pdf", "path/to/resume2.pdf"],
            job_description="path/to/jd.txt"  # or raw text string
        )
    """

    def __init__(self, config: ScoringConfig = None):
        self.config = config or ScoringConfig()
        self.scorer = ResumeScorer(self.config)

    def run(
        self,
        resumes: list[Union[str, Path, dict]],
        job_description: Union[str, Path],
        job_title: str = "the role",
        required_years: float = None,
    ) -> tuple:
        """
        Run full pipeline.

        Args:
            resumes:         list of file paths, raw text strings, or pre-processed dicts
            job_description: file path or raw text string for the JD
            job_title:       used in summary report
            required_years:  override for required years of experience

        Returns:
            (ranked_df, summary_dict)
        """
        logger.info(f"Starting ResumeIQ pipeline: {len(resumes)} resumes")

        # Process JD
        if isinstance(job_description, (str, Path)) and Path(str(job_description)).exists():
            jd_data = process_file(job_description)
        else:
            jd_data = process_text(str(job_description))

        logger.info(f"JD processed. Skills found: {len(jd_data.get('skills', set()))}")

        # Process resumes
        resume_data_list = []
        candidate_names = []

        for i, resume in enumerate(resumes):
            try:
                if isinstance(resume, dict):
                    # Pre-processed
                    data = resume
                    name = resume.get("name", f"Candidate {i+1}")
                elif isinstance(resume, (str, Path)) and Path(str(resume)).exists():
                    data = process_file(resume)
                    name = data.get("name", Path(str(resume)).stem)
                else:
                    # Raw text
                    data = process_text(str(resume))
                    name = f"Candidate {i+1}"

                resume_data_list.append(data)
                candidate_names.append(name)
                logger.debug(f"Processed {name}: {len(data.get('skills', set()))} skills")

            except Exception as e:
                logger.error(f"Failed to process resume {i}: {e}")
                # Add empty placeholder so ranking index stays consistent
                resume_data_list.append({})
                candidate_names.append(f"Candidate {i+1} (error)")

        # Score
        logger.info("Scoring candidates...")
        score_results = self.scorer.score_batch(resume_data_list, jd_data, required_years)

        # Rank
        ranked_df = rank_candidates(score_results, candidate_names, jd_data)
        summary = summary_report(ranked_df, job_title)

        logger.info(f"Pipeline complete. Top candidate: {summary.get('top_candidate')}")
        return ranked_df, summary
