import os
import lmstudio as lms

# Set host IP Windows tempat LM Studio berjalan
lms.configure_default_client("192.168.100.232:1234")

model = lms.llm(os.environ["LM_STUDIO_MODEL"])
prompt = "siapa itu muhammad mahruz zain jelaskan dia dosen politeknik caltex riau dosen prodi sistem informasi pakai bahasa indonesia dan jelaskan lebih spesifik prestasi nya"
result = model.respond(prompt)
print(result.content)
