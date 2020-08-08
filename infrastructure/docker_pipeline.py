from aws_cdk import (
    core,
    aws_codebuild as codebuild,
    aws_codecommit as codecommit,
    aws_codepipeline as codepipeline,
    aws_codepipeline_actions as codepipeline_actions,
    aws_ecr as ecr,
    aws_iam as iam,
)
from aws_cdk.aws_codecommit import Repository
from aws_cdk.aws_codebuild import BuildEnvironment, PipelineProject, LinuxBuildImage

from aws_cdk.aws_codepipeline import Artifact, Pipeline, StageProps
from aws_cdk.aws_codepipeline_actions import CodeCommitSourceAction, CodeBuildAction

repo_name = "ai-traineree"

class DockerPipelineStack(core.Stack):

    def __init__(self, scope: core.Construct, id: str, **kwargs):
        super().__init__(scope, id, *kwargs)



        ecr.Repository(self, "AiTrainereeRepo", repository_name='aitraineree')

        image_source_output = Artifact("ImageSource")
        image_repo = Repository.from_repository_name(
                self,
                "ImageRepo",
                repository_name=repo_name
        )

        image_build_project = self.create_image_build_project()

        code_commit_action = CodeCommitSourceAction(
            output=image_source_output,
            repository=image_repo,
            action_name="CodeCommit",
            branch="sagemaker",
        )

        code_build_action = CodeBuildAction(
            input=image_source_output,
            project=image_build_project,
            action_name="CodeBuid",
        )


        Pipeline(self, 'Pipeline',
            pipeline_name='DockerImagePipeline',
            stages=[StageProps(
                        stage_name="Source",
                        actions=[code_commit_action],
                    ), StageProps(
                        stage_name="BuildDocker",
                        actions=[code_build_action],
                    )],
        )

    def create_image_build_project(self):
        image_build = PipelineProject(self, "DockerImageBuild",
            environment=BuildEnvironment(
                build_image=LinuxBuildImage.STANDARD_3_0,
                privileged=True,
            )
        )

        image_build.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "ecr:BatchGetImage",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:CompleteLayerUpload",
                    "ecr:GetAuthorizationToken",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:UploadLayerPart",
                    "ecr:InitiateLayerUpload",
                    "ecr:PutImage",
                ],
                resources=["*"],
            )
        )
        return image_build

