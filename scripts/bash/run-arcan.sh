ARCAN_PATH=../../tools/arcan
REPOSITORY_PATH=../../data/raw/repositories
PROJECT=activej/activej
PROJECT_NAME="${PROJECT/\//-}"
OUTPATH=../../data/interim
LOGS_PATH=../../logs/arcan

mkdir -p $LOGS_PATH

$ARCAN_PATH/arcan.sh analyze -i $REPOSITORY_PATH/$PROJECT_NAME -p $PROJECT_NAME --remote https://github.com/$PROJECT \
              -o $OUTPATH -l JAVA -f $ARCAN_PATH/filters.yaml \
              -v output.writeDependencyGraph=true \
              output.writeAffected=false \
              output.writeComponentMetrics=False \
              output.writeSmellCharacteristics=False \
              metrics.componentMetrics=none \
              metrics.smellCharacteristics=none \
              metrics.indexCalculators=none \
              detectors.smellDetectors=none \
              -e --startDate 1-1-1 --endDate 2022-12-31 --intervalDays 28  2>&1 > $LOGS_PATH/$PROJECT_NAME.arcan.log