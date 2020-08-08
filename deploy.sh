AWS_ACCOUNT=$(aws sts get-caller-identity --output text | awk '{ print $1 }' )
AWS_REGION=us-west-2  # Change as necessary
REPO_NAME=aitraineree
REPO_TAG=latest
REPO=$REPO_NAME:$REPO_TAG

SAGEMAKER_ACCOUNT=763104351884

set -e
echo "Loging into $AWS_ACCOUNT account's ECR"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $SAGEMAKER_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com

echo "Building docker container with tag $REPO_NAME"
docker build --network host -t $REPO_NAME .

echo "docker tag $REPO $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO"
docker tag $REPO $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO

echo "docker push $AWS_ACCOUNT.dkr.ecr.AWS_REGION.amazonaws.com/$REPO"
docker push $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO

set +e