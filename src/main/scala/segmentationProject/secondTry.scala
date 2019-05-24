package segmentationProject

import java.awt.Color
import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.geometry._
import scalismo.io._
import scalismo.mesh.TriangleMesh
import scalismo.registration.{RigidTransformation, RotationTransform, TranslationTransform}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}
import scalismo.statisticalmodel.asm.{ActiveShapeModel, PreprocessedImage}
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.ui.api.{ScalismoUI, StatisticalMeshModelViewControls}
import scalismo.utils.{Memoize, Random}

import scala.io.StdIn


object secondTry {

  implicit val rng: Random = Random(42)

  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    val ui = ScalismoUI()

    val dataDir = "data/handedData/"

    val asm = ActiveShapeModelIO.readActiveShapeModel(new java.io.File(dataDir +
      "femur-asm.h5")).get
    val modelGroup = ui.createGroup("modelGroup")
    val modelView = ui.show(modelGroup, asm.statisticalModel, "shapeModel")
    modelView.meshView.color = Color.YELLOW

    val modelLms = LandmarkIO.readLandmarksJson[_3D](new java.io.File
    ("data/landmarks/model_transformed.json")).get
    val modelLmViews = ui.show(modelGroup, modelLms, "modelLandmarks")
    modelLmViews.foreach(lmView => lmView.color = java.awt.Color.BLUE)

    val profiles = asm.profiles
    profiles.foreach(profile => {
      val pointId = profile.pointId
      val distribution = profile.distribution
    })
    println("Loaded model.")

    val tests = Array(/*4, 14, 23, 25,*/ 30)
    val targets = Array(1, 9, 10, 13, 37)

//    tests.foreach { i: Int =>
//
//      val model = asm.statisticalModel
//
//      val image = ImageIO.read3DScalarImage[Short](new File(dataDir + "test/" + i + ".nii")).get
//        .map(_.toFloat)
//      val reference = MeshIO.readMesh(new File(dataDir + "test/" + i + ".stl")).get
//
//      val testGroup = ui.createGroup("test_" + i)
//      val imgView = ui.show(testGroup, image, "image")
//      val refView = ui.show(testGroup, reference, "reference")
//      refView.color = Color.GREEN
//
//      val imgLms = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/landmarks/" + i +
//        "_transformed.json")).get
//      val imgLmViews = ui.show(testGroup, imgLms, "imgLandmarks")
//      imgLmViews.foreach(lmView => lmView.color = java.awt.Color.RED)
//
//      val modelLmIds = modelLms.map(l => model.mean.pointSet.pointId(l.point).get)
//      val imgPoints = imgLms.map(l => l.point)
//
//      val preprocessedImage = asm.preprocessor(image)
//
//      val initialParameters = Parameters(EuclideanVector(0, 0, 0), (0.0, 0.0, 0.0),
//        DenseVector.zeros[Double](model.rank))
//
//      val logger = new Logger()
//
//      val initialSample: Sample = Sample("initial", initialParameters, computeCenterOfMass(model
//        .mean))
//      val bestSample = runChains(model, preprocessedImage, modelLmIds, imgPoints, initialSample,
//        logger, modelView, reference, asm, 1)
//      val bestFit = model.instance(bestSample.parameters.modelCoefficients)
//        .transform(bestSample.poseTransformation)
//      ui.show(testGroup, bestFit, "best fit")
//
//      println(logger.acceptanceRatios())
//      val dist = scalismo.mesh.MeshMetrics.avgDistance(bestFit, reference)
//      val hausDist = scalismo.mesh.MeshMetrics.hausdorffDistance(bestFit, reference)
//      println("Average Distance: " + dist)
//      println("Hausdorff Distance: " + hausDist)
//
//      MeshIO.writeMesh(bestFit, new File("data/mcmc/bestFit_" + i + ".stl"))
//
//      StdIn.readLine("Show next fit?")
//    }

    targets.foreach { i: Int =>
      val model = asm.statisticalModel

      val image = ImageIO.read3DScalarImage[Short](new File(dataDir + "targets/" + i + ".nii")).get
        .map(_.toFloat)

      val testGroup = ui.createGroup("target_" + i)
      val imgView = ui.show(testGroup, image, "image")

      val imgLms = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/landmarks/" + i +
        "_transformed.json")).get
      val imgLmViews = ui.show(testGroup, imgLms, "imgLandmarks")
      imgLmViews.foreach(lmView => lmView.color = java.awt.Color.RED)

      val modelLmIds = modelLms.map(l => model.mean.pointSet.pointId(l.point).get)
      val imgPoints = imgLms.map(l => l.point)

      val preprocessedImage = asm.preprocessor(image)

      val initialParameters = Parameters(EuclideanVector(0, 0, 0), (0.0, 0.0, 0.0),
        DenseVector.zeros[Double](model.rank))

      val logger = new Logger()

      val initialSample: Sample = Sample("initial", initialParameters, computeCenterOfMass(model
        .mean))

      val bestSample = runChains(model, preprocessedImage, modelLmIds, imgPoints, initialSample,
        logger, modelView, null, asm, 1)
      val bestFit = model.instance(bestSample.parameters.modelCoefficients)
        .transform(bestSample.poseTransformation)
      ui.show(testGroup, bestFit, "best fit")

      println(logger.acceptanceRatios())

      MeshIO.writeMesh(bestFit, new File("data/mcmc/bestFit_" + i + ".stl"))

//      StdIn.readLine("Show next fit?")

    }

    //    val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model,
    //      correspondences)
    //
    //    for ((id, _, _) <- newCorrespondences) {
    //      val meanPointPosition = computeMean(marginalizedModel, id)
    //      println(s"expected position for point at id $id  = $meanPointPosition")
    //      val cov = computeCovarianceFromSamples(marginalizedModel, id, meanPointPosition)
    //      println(s"posterior variance computed  for point at id (shape and pose) $id  = ${cov
    //      (0, 0)
    //      }, ${cov(1, 1)}, ${cov(2, 2)}")
    //    }

    //    val posteriorFixedPose = model.posterior(correspondences.toIndexedSeq)
    //
    //    for ((id, _, _) <- newCorrespondences) {
    //      val cov = posteriorFixedPose.cov(id, id)
    //      println(s"posterior variance computed by analytic posterior (shape only) for point
    //      with id " +
    //        s"$id = ${cov(0, 0)}, ${cov(1, 1)}, ${cov(2, 2)}")
    //    }
  }

  def runChains(model: StatisticalMeshModel, preprocessedImage: PreprocessedImage,
                modelLmIds: Seq[PointId], imgPoints: Seq[Point[_3D]], initial: Sample,
                logger: Logger, viewControls: StatisticalMeshModelViewControls,
                reference: TriangleMesh[_3D], asm: ActiveShapeModel, repetitions: Int): Sample = {

    val priorEvaluator = CachedEvaluator(PriorEvaluator(model))

    val likelihoodEvaluatorLM = getLandmarkLikelihoodEvaluator(model, modelLmIds, imgPoints)
    val posteriorEvaluatorLM = ProductEvaluator(priorEvaluator, likelihoodEvaluatorLM)

    val likelihoodEvaluatorASM = CachedEvaluator(ActiveShapeModelEvaluator(model, asm,
      preprocessedImage))

    val posteriorEvaluatorASM = ProductEvaluator(priorEvaluator, likelihoodEvaluatorASM)

    val shapeUpdateTinyProposal = ShapeUpdateProposal(model.rank, 0.02)
    val shapeUpdateSmallProposal = ShapeUpdateProposal(model.rank, 0.05)
    val shapeUpdateMediumProposal = ShapeUpdateProposal(model.rank, 0.1)
    val shapeUpdateLargeProposal = ShapeUpdateProposal(model.rank, 0.3)
    val rotationUpdateProposal = RotationUpdateProposal(0.01)
    val translationUpdateProposal = TranslationUpdateProposal(1.0)

    def runLMChain(initial: Sample): Sample = {

      val generator = MixtureProposal.fromProposalsWithTransition(
        (0.1, shapeUpdateLargeProposal), (0.2, shapeUpdateMediumProposal),
        (0.1, shapeUpdateSmallProposal), (0.3, rotationUpdateProposal),
        (0.3, translationUpdateProposal)
      )
      val samples = chain("Landmarks", model, initial, 5000, generator,
        posteriorEvaluatorLM, logger, viewControls, reference)

      samples.maxBy(posteriorEvaluatorASM.logValue)
    }

    def runASMChainLarge(initial: Sample): Sample = {

      val generator = MixtureProposal.fromProposalsWithTransition(
        (0.1, shapeUpdateLargeProposal), (0.4, shapeUpdateMediumProposal),
        (0.3, shapeUpdateSmallProposal), (0.1, rotationUpdateProposal),
        (0.1, translationUpdateProposal)
      )
      val samples = chain("Active Shape Model Large", model, initial, 5000, generator,
        posteriorEvaluatorASM, logger, viewControls, reference)

      samples.maxBy(posteriorEvaluatorASM.logValue)
    }

    def runASMChainSmall(initial: Sample): Sample = {

      val generator = MixtureProposal.fromProposalsWithTransition(
        (0.2, shapeUpdateMediumProposal), (0.5, shapeUpdateSmallProposal),
        (0.2, shapeUpdateTinyProposal), (0.05, rotationUpdateProposal),
        (0.05, translationUpdateProposal)
      )
      val samples = chain("Active Shape Model Small", model, initial, 5000, generator,
        posteriorEvaluatorASM, logger, viewControls, reference)

      samples.maxBy(posteriorEvaluatorASM.logValue)
    }

    var sample = initial
    for (_ <- 1 to repetitions) {
      sample = runLMChain(sample)
      println(logger.acceptanceRatios())
      sample = runASMChainLarge(sample)
      println(logger.acceptanceRatios())
      sample = runASMChainSmall(sample)
    }
    sample
  }

  def computeCenterOfMass(mesh: TriangleMesh[_3D]): Point[_3D] = {

    val normFactor = 1.0 / mesh.pointSet.numberOfPoints
    mesh.pointSet.points.foldLeft(Point(0, 0, 0))((sum, point) => sum + point.toVector *
      normFactor)
  }

  def getLandmarkLikelihoodEvaluator(model: StatisticalMeshModel, modelLMs: Seq[PointId],
                                     imgPoints: Seq[Point[_3D]]): CachedEvaluator[Sample] = {

    //landmark noise for tests
    val landmarkNoiseVariance = 9.0
    //landmark noise for targets
    //val landmarkNoiseVariance = 15.0
    val uncertainty = MultivariateNormalDistribution(DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * landmarkNoiseVariance)

    val correspondences = modelLMs.zip(imgPoints).map(modelIdWithTargetPoint => {
      val (modelId, targetPoint) = modelIdWithTargetPoint
      (modelId, targetPoint, uncertainty)
    })

    CachedEvaluator(CorrespondenceEvaluator(model, correspondences))
  }

  def chain(name: String, model: StatisticalMeshModel, initial: Sample, iterations: Int,
            generator: MixtureProposal[Sample] with TransitionProbability[Sample],
            evaluator: ProductEvaluator[Sample], logger: Logger,
            view: StatisticalMeshModelViewControls): IndexedSeq[Sample] = {

    chain(name, model, initial, iterations, generator, evaluator, logger, view, null)
  }

  def chain(name: String, model: StatisticalMeshModel, initial: Sample, iterations: Int,
            generator: MixtureProposal[Sample] with TransitionProbability[Sample],
            evaluator: ProductEvaluator[Sample], logger: Logger,
            view: StatisticalMeshModelViewControls, reference: TriangleMesh[_3D])
  : IndexedSeq[Sample] = {

    val chain: MetropolisHastings[Sample] = MetropolisHastings(generator, evaluator)
    val mhIterator = chain.iterator(initial, logger)

    val samplingIterator = for ((sample, iteration) <- mhIterator.zipWithIndex) yield {
      if (iteration % 100 == 0) println(name + ": iteration " + iteration)
      if (iteration % 500 == 0) {
        view.shapeModelTransformationView.shapeTransformationView.coefficients = sample
          .parameters.modelCoefficients
        view.shapeModelTransformationView.poseTransformationView.transformation = sample
          .poseTransformation

        if (reference != null) {
          val dist = scalismo.mesh.MeshMetrics.avgDistance(model.instance(sample.parameters

            .modelCoefficients).transform(sample
            .poseTransformation), reference)
          val hausDist = scalismo.mesh.MeshMetrics.hausdorffDistance(model.instance(sample
            .parameters.modelCoefficients).transform(sample
            .poseTransformation), reference)
          println("Average Distance: " + dist)
          println("Hausdorff Distance: " + hausDist)
        }
      }
      sample
    }

    samplingIterator.take(iterations).toIndexedSeq
  }

  //  def computeMean(model: StatisticalMeshModel, id: PointId, samples: Seq[Sample]): Point[_3D]
  //  = {
  //    var mean = EuclideanVector(0, 0, 0)
  //    for (sample <- samples) yield {
  //      val modelInstance = model.instance(sample.parameters.modelCoefficients)
  //      val pointForInstance = modelInstance.transform(sample.poseTransformation).pointSet
  //      .point(id)
  //      mean += pointForInstance.toVector
  //    }
  //    (mean * 1.0 / samples.size).toPoint
  //  }
  //
  //  def computeCovarianceFromSamples(model: StatisticalMeshModel, id: PointId, mean: Point[_3D],
  //                                   samples: Seq[Sample]): SquareMatrix[_3D] = {
  //
  //    var cov = SquareMatrix.zeros[_3D]
  //    for (sample <- samples) yield {
  //      val modelInstance = model.instance(sample.parameters.modelCoefficients)
  //      val pointForInstance = modelInstance.transform(sample.poseTransformation).pointSet
  //      .point(id)
  //      val v = pointForInstance - mean
  //      cov += v.outer(v)
  //    }
  //    cov * (1.0 / samples.size)
  //  }
}

case class Parameters(translationParameters: EuclideanVector[_3D],
                      rotationParameters: (Double, Double, Double),
                      modelCoefficients: DenseVector[Double])

case class Sample(generatedBy: String, parameters: Parameters, rotationCenter: Point[_3D]) {
  def poseTransformation: RigidTransformation[_3D] = {

    val translation = TranslationTransform(parameters.translationParameters)
    val rotation = RotationTransform(
      parameters.rotationParameters._1,
      parameters.rotationParameters._2,
      parameters.rotationParameters._3,
      rotationCenter
    )
    RigidTransformation(translation, rotation)
  }
}

case class PriorEvaluator(model: StatisticalMeshModel) extends DistributionEvaluator[Sample] {

  val translationPrior = breeze.stats.distributions.Gaussian(0.0, 5.0)
  val rotationPrior = breeze.stats.distributions.Gaussian(0, 0.1)

  override def logValue(sample: Sample): Double = {
    model.gp.logpdf(sample.parameters.modelCoefficients) +
      translationPrior.logPdf(sample.parameters.translationParameters.x) +
      translationPrior.logPdf(sample.parameters.translationParameters.y) +
      translationPrior.logPdf(sample.parameters.translationParameters.z) +
      rotationPrior.logPdf(sample.parameters.rotationParameters._1) +
      rotationPrior.logPdf(sample.parameters.rotationParameters._2) +
      rotationPrior.logPdf(sample.parameters.rotationParameters._3)
  }
}

case class ActiveShapeModelEvaluator(model: StatisticalMeshModel, asm: ActiveShapeModel,
                                     preprocessedImage: PreprocessedImage) extends
  DistributionEvaluator[Sample] {

  override def logValue(sample: Sample): Double = {

    val ids = asm.profiles.ids
    val mesh = model.instance(sample.parameters.modelCoefficients).transform(sample
      .poseTransformation)

    val likelihoods = for (id <- ids) yield {
      val profile = asm.profiles(id)
      val profilePointOnMesh = mesh.pointSet.point(profile.pointId)
      val featureAtPoint: DenseVector[Double] = {
        val featureAtPoint = asm.featureExtractor(preprocessedImage, profilePointOnMesh, mesh,
          profile.pointId)
        if (featureAtPoint.isDefined) featureAtPoint.get
        else DenseVector.zeros(11) // (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
      }
      profile.distribution.logpdf(featureAtPoint)
    }
    likelihoods.sum
  }
}

case class CorrespondenceEvaluator(model: StatisticalMeshModel, correspondences: Seq[
  (PointId, Point[_3D], MultivariateNormalDistribution)]) extends
  DistributionEvaluator[Sample] {

  val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model,
    correspondences)

  def marginalizeModelForCorrespondences(model: StatisticalMeshModel, correspondences: Seq[
    (PointId, Point[_3D], MultivariateNormalDistribution)]): (StatisticalMeshModel, Seq[
    (PointId, Point[_3D], MultivariateNormalDistribution)]) = {

    val (modelIds, _, _) = correspondences.unzip3
    val marginalizedModel = model.marginal(modelIds.toIndexedSeq)
    val newCorrespondences = correspondences.map(idWithTargetPoint => {
      val (id, targetPoint, uncertainty) = idWithTargetPoint
      val modelPoint = model.referenceMesh.pointSet.point(id)
      val newId = marginalizedModel.referenceMesh.pointSet.findClosestPoint(modelPoint).id
      (newId, targetPoint, uncertainty)
    })
    (marginalizedModel, newCorrespondences)
  }

  override def logValue(sample: Sample): Double = {

    val currModelInstance = marginalizedModel.instance(sample.parameters.modelCoefficients)
      .transform(sample.poseTransformation)

    val likelihoods = newCorrespondences.map(correspondence => {
      val (id, targetPoint, uncertainty) = correspondence
      val modelInstancePoint = currModelInstance.pointSet.point(id)
      val observedDeformation = targetPoint - modelInstancePoint

      uncertainty.logpdf(observedDeformation.toBreezeVector)
    })

    val loglikelihood = likelihoods.sum
    loglikelihood
  }
}

case class CachedEvaluator[A](evaluator: DistributionEvaluator[A]) extends
  DistributionEvaluator[A] {
  val memoizedLogValue = Memoize(evaluator.logValue, 10)

  override def logValue(sample: A): Double = {
    memoizedLogValue(sample)
  }
}

case class ShapeUpdateProposal(paramVectorSize: Int, stddev: Double) extends
  ProposalGenerator[Sample] with TransitionProbability[Sample] {

  val perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(paramVectorSize),
    DenseMatrix.eye[Double](paramVectorSize) * stddev * stddev)

  implicit val rng = Random(42)

  override def propose(sample: Sample): Sample = {

    val perturbation = perturbationDistr.sample()
    val newParameters = sample.parameters.copy(modelCoefficients = sample.parameters
      .modelCoefficients + perturbationDistr.sample)
    sample.copy(generatedBy = s"ShapeUpdateProposal ($stddev)", parameters = newParameters)
  }

  override def logTransitionProbability(from: Sample, to: Sample): Double = {

    val residual = to.parameters.modelCoefficients - from.parameters.modelCoefficients
    perturbationDistr.logpdf(residual)
  }
}

case class RotationUpdateProposal(stddev: Double) extends
  ProposalGenerator[Sample] with TransitionProbability[Sample] {

  val perturbationDistr = new MultivariateNormalDistribution(
    DenseVector.zeros[Double](3),
    DenseMatrix.eye[Double](3) * stddev * stddev)

  implicit val rng = Random(42)

  def propose(sample: Sample): Sample = {
    val perturbation = perturbationDistr.sample
    val newRotationParameters = (
      sample.parameters.rotationParameters._1 + perturbation(0),
      sample.parameters.rotationParameters._2 + perturbation(1),
      sample.parameters.rotationParameters._3 + perturbation(2)
    )
    val newParameters = sample.parameters.copy(rotationParameters = newRotationParameters)
    sample.copy(generatedBy = s"RotationUpdateProposal ($stddev)", parameters = newParameters)
  }

  override def logTransitionProbability(from: Sample, to: Sample): Double = {
    val residual = DenseVector(
      to.parameters.rotationParameters._1 - from.parameters.rotationParameters._1,
      to.parameters.rotationParameters._2 - from.parameters.rotationParameters._2,
      to.parameters.rotationParameters._3 - from.parameters.rotationParameters._3
    )
    perturbationDistr.logpdf(residual)
  }
}

case class TranslationUpdateProposal(stddev: Double) extends
  ProposalGenerator[Sample] with TransitionProbability[Sample] {

  val perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(3),
    DenseMatrix.eye[Double](3) * stddev * stddev)

  implicit val rng = Random(42)

  def propose(sample: Sample): Sample = {
    val newTranslationParameters = sample.parameters.translationParameters + EuclideanVector
      .fromBreezeVector(perturbationDistr.sample())
    val newParameters = sample.parameters.copy(translationParameters = newTranslationParameters)
    sample.copy(generatedBy = s"TranslationUpdateProposal ($stddev)", parameters = newParameters)
  }

  override def logTransitionProbability(from: Sample, to: Sample): Double = {
    val residual = to.parameters.translationParameters - from.parameters.translationParameters
    perturbationDistr.logpdf(residual.toBreezeVector)
  }
}

class Logger extends AcceptRejectLogger[Sample] {
  private val numAccepted = collection.mutable.Map[String, Int]()
  private val numRejected = collection.mutable.Map[String, Int]()

  override def accept(current: Sample, sample: Sample, generator: ProposalGenerator[Sample],
                      evaluator: DistributionEvaluator[Sample]): Unit = {

    val numAcceptedSoFar = numAccepted.getOrElseUpdate(sample.generatedBy, 0)
    numAccepted.update(sample.generatedBy, numAcceptedSoFar + 1)
  }

  override def reject(current: Sample, sample: Sample, generator: ProposalGenerator[Sample],
                      evaluator: DistributionEvaluator[Sample]): Unit = {

    val numRejectedSoFar = numRejected.getOrElseUpdate(sample.generatedBy, 0)
    numRejected.update(sample.generatedBy, numRejectedSoFar + 1)
  }

  def acceptanceRatios(): Map[String, Double] = {
    val generatorNames = numRejected.keys.toSet.union(numAccepted.keys.toSet)
    val acceptanceRatios = for (generatorName <- generatorNames) yield {
      val total = (numAccepted.getOrElse(generatorName, 0)
        + numRejected.getOrElse(generatorName, 0)).toDouble
      (generatorName, numAccepted.getOrElse(generatorName, 0) / total)
    }
    acceptanceRatios.toMap
  }
}

