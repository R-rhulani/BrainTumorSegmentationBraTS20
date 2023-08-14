import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  selectedFile: File | null = null;
  prediction: string | null = null;

  constructor(private http: HttpClient) {}

  onFileSelected(event: any) {
    this.selectedFile = event.target.files[0];
  }

  async uploadImage() {
    if (!this.selectedFile) {
      return;
    }

    const formData = new FormData();
    formData.append('image', this.selectedFile);

    try {
      const response: any = await this.http.post('/predict', formData).toPromise();
      this.prediction = JSON.stringify(response.prediction);
    } catch (error) {
      console.error(error);
      this.prediction = 'Error occurred while predicting.';
    }
  }
}
