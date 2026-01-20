from transformers import TrOCRProcessor, VisionEncoderDecoderModel

proc = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

proc.save_pretrained("trocr_printed")
model.save_pretrained("trocr_printed")


proc = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

proc.save_pretrained("trocr_handwritten")
model.save_pretrained("trocr_handwritten")
