from src.synopsis_processor import SynopsisProcessor

def run(video_path, fore_path, out_name, tube_save_path):
    sp = SynopsisProcessor(out_name)
    sp.getTubes(video_path, fore_path, tube_save_path)
    sp.rearranging()

if __name__ == "__main__":
    run(
        "D:/AAA_document_Lv/VideoEnrichment/dgcotr_new/ExpeData/original_video0201.mp4",
        "D:/AAA_document_Lv/VideoEnrichment/dgcotr_new/ExpeData/output_video0201.mp4",
        "video0202.avi",
        "video0202.txt"
    )
