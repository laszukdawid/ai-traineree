#!/usr/bin/env python3

from aws_cdk import core

from cdk.cdk_stack import CdkStack

from docker_pipeline import DockerPipelineStack


app = core.App()
DockerPipelineStack(app, "DockerPipelineStack")

app.synth()
