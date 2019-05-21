package segmentationProject

import java.awt.Color
import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.geometry._
import scalismo.io._
import scalismo.mesh.TriangleMesh
import scalismo.registration.{
  LandmarkRegistration, RigidTransformation, RotationTransform,
  TranslationTransform
}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}
import scalismo.statisticalmodel.asm.{ActiveShapeModel, PreprocessedImage}
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.ui.api.{ScalismoUI, StatisticalMeshModelViewControls}
import scalismo.utils.{Memoize, Random}

import scala.collection.immutable
import scala.io.StdIn


object secondTry {

  implicit val rng: Random = scalismo.utils.Random(42)

  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    val ui = ScalismoUI()

    val dataDir = "data/handedData/"

    val asm = ActiveShapeModelIO.readActiveShapeModel(new java.io.File(dataDir +
      "femur-asm.h5")).get
    val modelGroup = ui.createGroup("modelGroup")
    val modelView = ui.show(modelGroup, asm.statisticalModel, "shapeModel")

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

    val tests = Array(4, 14, 23, 25, 30)
    val targets = Array(1, 9, 10, 13, 37)

    tests.foreach { i: Int =>

      val model = asm.statisticalModel

      val image = ImageIO.read3DScalarImage[Short](new File(dataDir + "test/" + i + ".nii")).get
        .map(_.toFloat)
      val reference = MeshIO.readMesh(new File(dataDir + "test/" + i + ".stl")).get

      val testGroup = ui.createGroup("test_" + i)
      val imgView = ui.show(testGroup, image, "image")
      val refView = ui.show(testGroup, reference, "reference")
      refView.color = Color.GREEN

      val imgLms = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/landmarks/" + i +
        "_transformed.json")).get
      val imgLmViews = ui.show(testGroup, imgLms, "imgLandmarks")
      imgLmViews.foreach(lmView => lmView.color = java.awt.Color.RED)

      val modelLmIds = modelLms.map(l => model.mean.pointSet.pointId(l.point).get)
      val modelLmPoints = modelLmIds.map(id => model.mean.pointSet.point(id))
      val imgPoints = imgLms.map(l => l.point)

      val preprocessedImage = asm.preprocessor(image)

      val priorEvaluator = CachedEvaluator(PriorEvaluator(model))

      val likelihoodEvaluatorLM = getLandmarkLikelihoodEvaluator(model, modelLmIds, imgPoints)
      val posteriorEvaluatorLM = ProductEvaluator(priorEvaluator, likelihoodEvaluatorLM)

      val likelihoodEvaluatorASM = CachedEvaluator(ActiveShapeModelEvaluator(model, asm,
        preprocessedImage))
      val posteriorEvaluatorASM = ProductEvaluator(priorEvaluator, likelihoodEvaluatorASM)

      val shapeUpdateSmallProposal = ShapeUpdateProposal(model.rank, 0.1)
      val shapeUpdateLargeProposal = ShapeUpdateProposal(model.rank, 1)
      val rotationUpdateProposal = RotationUpdateProposal(0.01)
      val translationUpdateProposal = TranslationUpdateProposal(1.0)

      val initialParameters = Parameters(EuclideanVector(0, 0, 0), (0.0, 0.0, 0.0),
        DenseVector.zeros[Double](model.rank))

      val logger = new Logger()

      //      val landmarks = modelLmPoints.zip(imgPoints)
      //      val bestTransform: RigidTransformation[_3D] = LandmarkRegistration
      //        .rigid3DLandmarkRegistration(landmarks, center = Point(0, 0, 0))
      //      val alignedModel = model.transform(bestTransform)
      //
      //      val alignedGroup = ui.createGroup("aligned")
      //      val alignedView = ui.show(alignedGroup, alignedModel, "aligned")
      //      alignedView.meshView.color = Color.BLUE
      //
      //      val initialSampleASM = Sample("initial", initialParameters, computeCenterOfMass
      //      (alignedModel.mean))

      val initialSample: Sample = Sample("initial", initialParameters, computeCenterOfMass(model
        .mean))
      val generatorLM = MixtureProposal.fromProposalsWithTransition(
        (0.2, shapeUpdateLargeProposal), (0.2, shapeUpdateSmallProposal),
        (0.3, rotationUpdateProposal), (0.3, translationUpdateProposal)
      )
      val samplesLM = chain("Landmarks", model, initialSample, 5000, generatorLM,
        posteriorEvaluatorLM, logger, modelView, reference)

      val initialSampleASM = samplesLM.maxBy(posteriorEvaluatorLM.logValue)
      val lmView = ui.show(testGroup, model.instance(initialSampleASM.parameters
        .modelCoefficients).transform(initialSampleASM.poseTransformation),
        "after Landmark alignment")
      lmView.color = Color.YELLOW
      val generatorASM = MixtureProposal.fromProposalsWithTransition(
        (0.4, shapeUpdateLargeProposal), (0.2, shapeUpdateSmallProposal),
        (0.2, rotationUpdateProposal), (0.2, translationUpdateProposal)
      )
      val samplesASM = chain("Active Shape Model Large", model, initialSampleASM, 5000,
        generatorASM, posteriorEvaluatorASM, logger, modelView, reference)

      val initialSampleASM2 = samplesASM.maxBy(posteriorEvaluatorASM.logValue)
      val asmView = ui.show(testGroup, model.instance(initialSampleASM2.parameters
        .modelCoefficients).transform(initialSampleASM2.poseTransformation),
        "after first ASM")
      asmView.color = Color.RED
      val generatorASM2 = MixtureProposal.fromProposalsWithTransition(
        (0.1, shapeUpdateLargeProposal), (0.5, shapeUpdateSmallProposal),
        (0.2, rotationUpdateProposal), (0.2, translationUpdateProposal)
      )
      val samplesASM2 = chain("Active Shape Model Small", model, initialSampleASM2, 5000,
        generatorASM2, posteriorEvaluatorASM, logger, modelView, reference)

      // Why does overall result get worse if I increase numbers of iterations?
      // 15000 -> 5000 -> 8000 ----> avg=0.62, hdrf=6.2

      println(logger.acceptanceRatios())

      val bestSample = samplesASM2.maxBy(posteriorEvaluatorASM.logValue)
      val bestFit = model.instance(bestSample.parameters.modelCoefficients).transform(bestSample
        .poseTransformation)
      ui.show(testGroup, bestFit, "best fit")

      val dist = scalismo.mesh.MeshMetrics.avgDistance(bestFit, reference)
      val hausDist = scalismo.mesh.MeshMetrics.hausdorffDistance(bestFit, reference)
      println("Average Distance: " + dist)
      println("Hausdorff Distance: " + hausDist)

      MeshIO.writeMesh(bestFit, new File("data/mcmc/bestFit_" + i + ".stl"))

      StdIn.readLine("Show next fit?")
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

  def computeCenterOfMass(mesh: TriangleMesh[_3D]): Point[_3D] = {
    val normFactor = 1.0 / mesh.pointSet.numberOfPoints
    mesh.pointSet.points.foldLeft(Point(0, 0, 0))((sum, point) => sum + point.toVector *
      normFactor)
  }

  def getSamplingIterator(name: String, iterator: Iterator[Sample], model: StatisticalMeshModel,
                          view: StatisticalMeshModelViewControls): Iterator[Sample] = {

    for ((sample, iteration) <- iterator.zipWithIndex) yield {
      println(name + ": iteration " + iteration)
      if (iteration % 500 == 0) {
        view.shapeModelTransformationView.shapeTransformationView.coefficients = sample
          .parameters.modelCoefficients
        view.shapeModelTransformationView.poseTransformationView.transformation = sample
          .poseTransformation

        //        val dist = scalismo.mesh.MeshMetrics.avgDistance(model.instance(sample.parameters
        //          .modelCoefficients).transform(sample
        //          .poseTransformation), reference)
        //        val hausDist = scalismo.mesh.MeshMetrics.hausdorffDistance(model.instance(sample
        //          .parameters.modelCoefficients).transform(sample
        //          .poseTransformation), reference)
        //        println("Average Distance: " + dist)
        //        println("Hausdorff Distance: " + hausDist)
      }
      sample
    }
  }

  def getLandmarkLikelihoodEvaluator(model: StatisticalMeshModel, modelLMs: Seq[PointId],
                                     imgPoints: Seq[Point[_3D]]): CachedEvaluator[Sample] = {

    val landmarkNoiseVariance = 9.0
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
            posteriorEvaluator: ProductEvaluator[Sample], logger: Logger,
            view: StatisticalMeshModelViewControls): IndexedSeq[Sample] = {

    chain(name, model, initial, iterations, generator, posteriorEvaluator, logger, view, null)
  }

  def chain(name: String, model: StatisticalMeshModel, initial: Sample, iterations: Int,
            generator: MixtureProposal[Sample] with TransitionProbability[Sample],
            posteriorEvaluator: ProductEvaluator[Sample], logger: Logger,
            view: StatisticalMeshModelViewControls, reference: TriangleMesh[_3D])
  : IndexedSeq[Sample] = {

    val chain: MetropolisHastings[Sample] = MetropolisHastings(generator, posteriorEvaluator)
    val mhIterator = chain.iterator(initial, logger)

    val samplingIterator = for ((sample, iteration) <- mhIterator.zipWithIndex) yield {
      println(name + ": iteration " + iteration)
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

  def performChaines() = {

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
        else DenseVector(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
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

  val perturbationDistr = new MultivariateNormalDistribution(
    DenseVector.zeros(paramVectorSize),
    DenseMatrix.eye[Double](paramVectorSize) * stddev * stddev
  )

  implicit val rng = scalismo.utils.Random(42)

  override def propose(sample: Sample): Sample = {
    val perturbation = perturbationDistr.sample()
    val newParameters = sample.parameters.copy(modelCoefficients = sample.parameters
      .modelCoefficients + perturbationDistr.sample)
    sample.copy(generatedBy = s"ShapeUpdateProposal ($stddev)", parameters = newParameters)
  }

  override def logTransitionProbability(from: Sample, to: Sample) = {
    val residual = to.parameters.modelCoefficients - from.parameters.modelCoefficients
    perturbationDistr.logpdf(residual)
  }
}

case class RotationUpdateProposal(stddev: Double) extends
  ProposalGenerator[Sample] with TransitionProbability[Sample] {

  val perturbationDistr = new MultivariateNormalDistribution(
    DenseVector.zeros[Double](3),
    DenseMatrix.eye[Double](3) * stddev * stddev)

  implicit val rng = scalismo.utils.Random(42)

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

  override def logTransitionProbability(from: Sample, to: Sample) = {
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

  implicit val rng = scalismo.utils.Random(42)

  def propose(sample: Sample): Sample = {
    val newTranslationParameters = sample.parameters.translationParameters + EuclideanVector
      .fromBreezeVector(perturbationDistr.sample())
    val newParameters = sample.parameters.copy(translationParameters = newTranslationParameters)
    sample.copy(generatedBy = s"TranlationUpdateProposal ($stddev)", parameters = newParameters)
  }

  override def logTransitionProbability(from: Sample, to: Sample) = {
    val residual = to.parameters.translationParameters - from.parameters.translationParameters
    perturbationDistr.logpdf(residual.toBreezeVector)
  }
}

class Logger extends AcceptRejectLogger[Sample] {
  private val numAccepted = collection.mutable.Map[String, Int]()
  private val numRejected = collection.mutable.Map[String, Int]()

  override def accept(current: Sample,
                      sample: Sample,
                      generator: ProposalGenerator[Sample],
                      evaluator: DistributionEvaluator[Sample]
                     ): Unit = {
    val numAcceptedSoFar = numAccepted.getOrElseUpdate(sample.generatedBy, 0)
    numAccepted.update(sample.generatedBy, numAcceptedSoFar + 1)
  }

  override def reject(current: Sample,
                      sample: Sample,
                      generator: ProposalGenerator[Sample],
                      evaluator: DistributionEvaluator[Sample]
                     ): Unit = {
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

/*
Active Shape Model Large: iteration 947
Active Shape Model Large: iteration 948
Exception in thread "main" java.util.NoSuchElementException: None.get
at scala.None$.get(Option.scala:349)
at scala.None$.get(Option.scala:347)
at segmentationProject.ActiveShapeModelEvaluator.$anonfun$logValue$1(secondTry.scala:334)
at segmentationProject.ActiveShapeModelEvaluator.$anonfun$logValue$1$adapted(secondTry.scala:330)
at scala.collection.TraversableLike.$anonfun$map$1(TraversableLike.scala:234)
at scala.collection.Iterator.foreach(Iterator.scala:944)
at scala.collection.Iterator.foreach$(Iterator.scala:944)
at scala.collection.AbstractIterator.foreach(Iterator.scala:1432)
at scala.collection.IterableLike.foreach(IterableLike.scala:71)
at scala.collection.IterableLike.foreach$(IterableLike.scala:70)
at scala.collection.AbstractIterable.foreach(Iterable.scala:54)
at scala.collection.TraversableLike.map(TraversableLike.scala:234)
at scala.collection.TraversableLike.map$(TraversableLike.scala:227)
at scala.collection.AbstractTraversable.map(Traversable.scala:104)
at segmentationProject.ActiveShapeModelEvaluator.logValue(secondTry.scala:330)
at segmentationProject.ActiveShapeModelEvaluator.logValue(secondTry.scala:320)
at segmentationProject.CachedEvaluator.$anonfun$memoizedLogValue$1(secondTry.scala:383)
at segmentationProject.CachedEvaluator.$anonfun$memoizedLogValue$1$adapted(secondTry.scala:383)
at scalismo.utils.Memoize.$anonfun$apply$1(Memoize.scala:59)
at scalismo.utils.Memoize$Holder.getOrPut(Memoize.scala:32)
at scalismo.utils.Memoize.apply(Memoize.scala:59)
at segmentationProject.CachedEvaluator.logValue(secondTry.scala:386)
at scalismo.sampling.evaluators.ProductEvaluator.$anonfun$logValue$1(ProductEvaluator.scala:29)
at scalismo.sampling.evaluators.ProductEvaluator.$anonfun$logValue$1$adapted(ProductEvaluator
.scala:29)
at scala.collection.Iterator$$anon$10.next(Iterator.scala:457)
at scala.collection.Iterator.foreach(Iterator.scala:944)
at scala.collection.Iterator.foreach$(Iterator.scala:944)
at scala.collection.AbstractIterator.foreach(Iterator.scala:1432)
at scala.collection.TraversableOnce.foldLeft(TraversableOnce.scala:157)
at scala.collection.TraversableOnce.foldLeft$(TraversableOnce.scala:155)
at scala.collection.AbstractIterator.foldLeft(Iterator.scala:1432)
at scala.collection.TraversableOnce.sum(TraversableOnce.scala:216)
at scala.collection.TraversableOnce.sum$(TraversableOnce.scala:216)
at scala.collection.AbstractIterator.sum(Iterator.scala:1432)
at scalismo.sampling.evaluators.ProductEvaluator.logValue(ProductEvaluator.scala:29)
at scalismo.sampling.algorithms.MetropolisHastings.next(Metropolis.scala:80)
at scalismo.sampling.algorithms.MetropolisHastings.$anonfun$iterator$2(Metropolis.scala:72)
at scala.collection.Iterator$$anon$7.next(Iterator.scala:137)
at scala.collection.Iterator$$anon$20.next(Iterator.scala:889)
at scala.collection.Iterator$$anon$20.next(Iterator.scala:885)
at scala.collection.Iterator$$anon$12.hasNext(Iterator.scala:510)
at scala.collection.Iterator$$anon$10.hasNext(Iterator.scala:456)
at scala.collection.Iterator$SliceIterator.hasNext(Iterator.scala:263)
at scala.collection.Iterator.foreach(Iterator.scala:944)
at scala.collection.Iterator.foreach$(Iterator.scala:944)
at scala.collection.AbstractIterator.foreach(Iterator.scala:1432)
at scala.collection.generic.Growable.$plus$plus$eq(Growable.scala:59)
at scala.collection.generic.Growable.$plus$plus$eq$(Growable.scala:50)
at scala.collection.immutable.VectorBuilder.$plus$plus$eq(Vector.scala:658)
at scala.collection.immutable.VectorBuilder.$plus$plus$eq(Vector.scala:635)
at scala.collection.TraversableOnce.to(TraversableOnce.scala:310)
at scala.collection.TraversableOnce.to$(TraversableOnce.scala:308)
at scala.collection.AbstractIterator.to(Iterator.scala:1432)
at scala.collection.TraversableOnce.toIndexedSeq(TraversableOnce.scala:300)
at scala.collection.TraversableOnce.toIndexedSeq$(TraversableOnce.scala:300)
at scala.collection.AbstractIterator.toIndexedSeq(Iterator.scala:1432)
at segmentationProject.secondTry$.chain(secondTry.scala:251)
at segmentationProject.secondTry$.$anonfun$main$3(secondTry.scala:114)
at segmentationProject.secondTry$.$anonfun$main$3$adapted(secondTry.scala:57)
at scala.collection.IndexedSeqOptimized.foreach(IndexedSeqOptimized.scala:32)
at scala.collection.IndexedSeqOptimized.foreach$(IndexedSeqOptimized.scala:29)
at scala.collection.mutable.ArrayOps$ofInt.foreach(ArrayOps.scala:242)
at segmentationProject.secondTry$.main(secondTry.scala:57)
at segmentationProject.secondTry.main(secondTry.scala)
*/
