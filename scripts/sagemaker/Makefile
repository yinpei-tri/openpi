# Standalone build/push targets for the SageMaker training image.
# Defaults match scripts/sagemaker/config.yaml — keep them in sync.
#
# Run from the repo root:
#   make -C scripts/sagemaker docker-build-sm
#   make -C scripts/sagemaker docker-push-sm

REPO_NAME       ?= yinpeidai-openpi-train
PROFILE         ?= sagemaker
REGION          ?= us-west-2
DLC_ACCOUNT     ?= 763104351884
ROOT            := $(realpath ../..)

.PHONY: docker-build-sm docker-push-sm docker-shell

docker-build-sm:
	@account=$$(aws sts get-caller-identity --query Account --output text --profile $(PROFILE)) && \
	fullname=$$account.dkr.ecr.$(REGION).amazonaws.com/$(REPO_NAME):latest && \
	echo "Building $$fullname (context=$(ROOT))" && \
	aws ecr get-login-password --region $(REGION) --profile $(PROFILE) | \
	    docker login --username AWS --password-stdin $(DLC_ACCOUNT).dkr.ecr.$(REGION).amazonaws.com && \
	docker build --progress=plain \
	    -f $(ROOT)/scripts/sagemaker/train.Dockerfile \
	    --build-arg AWS_REGION=$(REGION) \
	    -t $(REPO_NAME) $(ROOT)

docker-push-sm: docker-build-sm
	@account=$$(aws sts get-caller-identity --query Account --output text --profile $(PROFILE)) && \
	fullname=$$account.dkr.ecr.$(REGION).amazonaws.com/$(REPO_NAME):latest && \
	aws ecr describe-repositories --repository-names $(REPO_NAME) --region $(REGION) --profile $(PROFILE) --no-cli-pager > /dev/null 2>&1 || \
	    aws ecr create-repository --repository-name $(REPO_NAME) --region $(REGION) --profile $(PROFILE) --no-cli-pager && \
	aws ecr get-login-password --region $(REGION) --profile $(PROFILE) | \
	    docker login --username AWS --password-stdin $$fullname && \
	docker tag $(REPO_NAME) $$fullname && \
	docker push $$fullname

# Local sanity check — runs the container without SageMaker channels mounted.
# Inside the container, run `python -c "import openpi"` to confirm the source
# baked in correctly. Won't actually train (no data).
docker-shell:
	docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
	    --entrypoint /bin/bash \
	    $(REPO_NAME)
